#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
"""
run_flagaudio_tests.py

Python replacement of your original shell script.
Saves per-operator logs and produces summary.json + summary.xlsx.

Usage:
    python run_flagaudio_tests.py --flagaudio /path/to/flagaudio --op-list /path/to/ops.txt --gpus 0,1,2,3
"""

import os
import re
import json
import argparse
import shutil
import subprocess
import datetime
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openpyxl import Workbook
from decimal import Decimal, getcontext

# increase decimal precision
getcontext().prec = 18

# ---------------- Global incremental summary ----------------

SUMMARY_LOCK = threading.Lock()
GLOBAL_RESULTS = {}

# ---------------- Helpers ----------------

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def run_cmd_capture(cmd, cwd=None, env=None):
    print(f"[INFO] CMD: {cmd}  (cwd={cwd})")
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = p.communicate()
    return out or "", err or "", p.returncode

# robust numeric validator
NUM_RE = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')

def is_number(s):
    return bool(NUM_RE.match(s.strip()))

def to_decimal(s):
    s2 = s.strip()
    if not is_number(s2):
        raise ValueError(f"Not numeric: {s}")
    return Decimal(s2)

# ---------------- Parse pytest summary ----------------

ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def parse_pytest_summary_from_text(text):
    """
    Return: passed, failed, skipped, errors, total
    """
    clean = ANSI_RE.sub('', text)
    counters = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

    for m in re.finditer(r'(\d+)\s+([A-Za-z_]+)', clean):
        num = int(m.group(1))
        key = m.group(2).lower()
        if key in counters:
            counters[key] = num

    passed = counters["passed"]
    failed = counters["failed"]
    skipped = counters["skipped"]
    errors = counters["errors"]
    total = passed + failed + skipped
    return passed, failed, skipped, errors, total

# ---------------- Per-operator workflow ----------------

def run_accuracy(op, gpu_id, flagaudio_path, op_dir):
    print(f"[INFO][GPU {gpu_id}] Starting accuracy for '{op}'")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    no_cpu_list = ["get_scheduler_metadata", "grouped_topk", "per_token_group_quant_fp8", "flash_attention_forward"]
    if f'{op}' in no_cpu_list:
        cmd = f'pytest -m "{op}" -vs'
    else:
        cmd = f'pytest -m "{op}" --ref cpu -vs'
    out, err, code = run_cmd_capture(
        cmd,
        cwd=os.path.join(flagaudio_path, "tests"),
        env=env
    )

    combined = out + "\n" + err
    acc_log = os.path.join(op_dir, "accuracy.log")
    with open(acc_log, "w") as f:
        f.write(combined)

    passed, failed, skipped, errors, total = parse_pytest_summary_from_text(combined)

    # ✅ 新 status 规则
    if failed > 0:
        status = "FAIL"
    elif errors > 0 and total == 0:
        status = "FAIL"  # pytest 没跑起来
    elif passed == 0:
        status = "FAIL"  # 实际没有执行测例
    else:
        status = "PASS"

    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "total": total,
        "status": status,
        "log_path": acc_log,
        "exit_code": code
    }

def run_benchmark_and_parse(op, gpu_id, flagaudio_path, op_dir):
    print(f"[INFO][GPU {gpu_id}] Starting benchmark for '{op}'")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    benchmark_dir = os.path.join(flagaudio_path, "benchmark")
    ensure_dir(benchmark_dir)

    pattern = f"result-m_{op}--level_core--record_log.log"
    for p in Path(benchmark_dir).glob(pattern):
        try:
            p.unlink()
        except Exception:
            pass

    cmd = f'pytest -m "{op}" --level core --record log'
    out, err, code = run_cmd_capture(cmd, cwd=benchmark_dir, env=env)

    perf_console_log = os.path.join(op_dir, "perf.log")
    with open(perf_console_log, "w") as f:
        f.write(out + "\n" + err)

    perf_result_file = None
    for p in Path(benchmark_dir).glob(pattern):
        perf_result_file = str(p)
        break

    if not perf_result_file:
        return {
            "status": "NO_RESULT",
            "perf_console_log": perf_console_log,
            "perf_result_file": None,
            "parsed_summary": None,
            "performance_rows": []
        }

    dest = os.path.join(op_dir, os.path.basename(perf_result_file))
    shutil.move(perf_result_file, dest)
    perf_result_file = dest

    with open(perf_result_file) as f:
        lines = f.readlines()
    seen = set()
    uniq = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    with open(perf_result_file, "w") as f:
        f.writelines(uniq)

    parsed_summary_path = os.path.join(op_dir, "parsed_summary.log")
    try:
        cmd = f'python3 summary_for_plot.py "{perf_result_file}"'
        out2, err2, _ = run_cmd_capture(cmd, cwd=benchmark_dir)
        combined2 = out2 + "\n" + err2

        with open(parsed_summary_path, "w") as f:
            f.write(combined2)
    except Exception:
        pass

    performance_rows = []
    try:
        with open(parsed_summary_path) as f:
            lines = f.readlines()

        start = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith("XCCL"):
                continue
            if line.lower().startswith("op_name"):
                start = True
                continue
            if not start:
                continue

            cols = re.split(r'\s+', line)
            if len(cols) < 8:
                continue

            row = {
                "func_name": cols[0],
                "float16": cols[1],
                "float32": cols[2],
                "bfloat16": cols[3],
                "int16": cols[4],
                "int32": cols[5],
                "bool": cols[6],
                "cfloat": cols[7],
            }

            vals = []
            for v in row.values():
                try:
                    dv = to_decimal(v)
                    if dv > 0:
                        vals.append(dv)
                except Exception:
                    pass

            row["avg_speedup"] = str(round(float(sum(vals) / len(vals)), 6)) if vals else "0"
            performance_rows.append(row)

    except Exception:
        pass

    return {
        "status": "OK",
        "perf_console_log": perf_console_log,
        "perf_result_file": perf_result_file,
        "parsed_summary": parsed_summary_path,
        "performance_rows": performance_rows
    }

# ---------------- Worker ----------------

def worker_process_ops(gpu_id, ops_list, flagaudio_path, results_dir):
    for op in ops_list:
        op = op.strip()
        if not op:
            continue

        op_dir = os.path.join(results_dir, op)
        ensure_dir(op_dir)

        acc = run_accuracy(op, gpu_id, flagaudio_path, op_dir)
        perf = run_benchmark_and_parse(op, gpu_id, flagaudio_path, op_dir)

        with SUMMARY_LOCK:
            GLOBAL_RESULTS[op] = {
                "gpu": gpu_id,
                "accuracy": acc,
                "performance": perf
            }
            write_summary_json_and_xlsx(GLOBAL_RESULTS, results_dir)

# ---------------- Summary writer ----------------

def write_summary_json_and_xlsx(summary_map, results_dir):
    json_path = os.path.join(results_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump([
            {
                "operator": op,
                "accuracy": info["accuracy"],
                "performance": info["performance"]["performance_rows"]
            }
            for op, info in summary_map.items()
        ], f, indent=2)

    xlsx_path = os.path.join(results_dir, "summary.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"

    ws.append([
        "operator", "acc_status", "passed", "failed", "skipped", "errors", "total", "acc_exit_code",
        "func_name", "avg_speedup",
        "float16", "float32", "bfloat16", "int16", "int32", "bool", "cfloat",
        "perf_status", "perf_console_log", "perf_result_file", "parsed_summary"
    ])

    for op, info in summary_map.items():
        acc = info["accuracy"]
        perf = info["performance"]
        rows = perf["performance_rows"] or [{}]
        first = True

        for r in rows:
            ws.append([
                op if first else "",
                acc["status"] if first else "",
                acc["passed"] if first else "",
                acc["failed"] if first else "",
                acc["skipped"] if first else "",
                acc["errors"] if first else "",
                acc["total"] if first else "",
                acc["exit_code"] if first else "",
                r.get("func_name", ""),
                r.get("avg_speedup", ""),
                r.get("float16", ""),
                r.get("float32", ""),
                r.get("bfloat16", ""),
                r.get("int16", ""),
                r.get("int32", ""),
                r.get("bool", ""),
                r.get("cfloat", ""),
                perf["status"],
                perf["perf_console_log"],
                perf["perf_result_file"],
                perf["parsed_summary"]
            ])
            first = False

    wb.save(xlsx_path)

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagaudio", required=True)
    parser.add_argument("--op-list", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    with open(args.op_list) as f:
        ops = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    results_dir = args.results_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results_{now_ts()}"
    )
    ensure_dir(results_dir)

    tasks = {g: [] for g in gpus}
    for i, op in enumerate(ops):
        tasks[gpus[i % len(gpus)]].append(op)

    with ThreadPoolExecutor(max_workers=len(gpus)) as exe:
        futures = []
        for g in gpus:
            if tasks[g]:
                futures.append(
                    exe.submit(worker_process_ops, g, tasks[g], args.flagaudio, results_dir)
                )
        for f in as_completed(futures):
            f.result()

    print("[INFO] All done.")

if __name__ == "__main__":
    main()
