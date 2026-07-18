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

"""
算子测试结果分析工具
用法: python analyze_ops.py <结果文件夹路径>
输出: 在目标文件夹下生成 report.html
"""

import sys
import os
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("错误: 需要安装依赖库")
    print("请运行: pip install pandas numpy openpyxl")
    sys.exit(1)


def read_log_file(folder, op_name, log_type="accuracy"):
    """读取算子的日志文件"""
    log_path = folder / op_name / f"{log_type}.log"
    if log_path.exists():
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return "无法读取日志文件"
    return "日志文件不存在"


def analyze_data(df, folder=None):
    """分析数据并返回统计结果"""
    # 算子数量（有 operator 名称的行）
    total = df['operator'].notna().sum()

    # 精度正确: passed > 0 且 failed == 0 且 errors == 0
    df['accuracy_pass'] = (df['passed'] > 0) & (df['failed'] == 0) & (df['errors'] == 0)

    # 有性能结果
    df['has_perf'] = df['avg_speedup'].notna() & (df['avg_speedup'] > 0)

    # 成功: 精度正确且有性能结果
    df['success'] = df['accuracy_pass'] & df['has_perf']

    success_df = df[df['success']]

    # 用于计算中位数/平均值的加速比（与 Excel 一致）
    # 对于 operator 为空的行，继承前一个 operator 的正确性
    df['operator_filled'] = df['operator'].ffill()
    # 构建每个 operator 的正确性映射
    accuracy_map = df[df['operator'].notna()].set_index('operator')['accuracy_pass'].to_dict()
    # 每行根据其所属 operator 判断正确性
    df['inherited_accuracy'] = df['operator_filled'].map(accuracy_map).fillna(False)
    # 参与统计: 所属算子精度正确 且 有加速比结果
    df['include_in_stats'] = df['inherited_accuracy'] & df['has_perf']
    speedups = df[df['include_in_stats']]['avg_speedup'].values if df['include_in_stats'].any() else np.array([])

    # 失败分类
    no_accuracy = df[(df['passed'] == 0) & (df['failed'] == 0) & (df['skipped'] == 0) & (df['errors'] == 0)]
    accuracy_failed = df[(df['failed'] > 0) | (df['errors'] > 0)]
    no_perf = df[df['accuracy_pass'] & ~df['has_perf']]

    # 加速比分布
    if len(speedups) > 0:
        below_08 = int(np.sum(speedups < 0.8))
        between_08_1 = int(np.sum((speedups >= 0.8) & (speedups <= 1.0)))
        above_1 = int(np.sum(speedups > 1.0))
        total_speedups = len(speedups)
    else:
        below_08 = between_08_1 = above_1 = total_speedups = 0

    return {
        'total': total,
        'success_count': int(df['success'].sum()),
        'no_accuracy_count': len(no_accuracy),
        'accuracy_failed_count': len(accuracy_failed),
        'no_perf_count': len(no_perf),
        'speedups': speedups,
        'median': float(np.median(speedups)) if len(speedups) > 0 else 0,
        'mean': float(np.mean(speedups)) if len(speedups) > 0 else 0,
        'min': float(np.min(speedups)) if len(speedups) > 0 else 0,
        'max': float(np.max(speedups)) if len(speedups) > 0 else 0,
        'below_08': below_08,
        'between_08_1': between_08_1,
        'above_1': above_1,
        'total_speedups': total_speedups,
        'slow_ops': success_df[success_df['avg_speedup'] < 0.8].sort_values('avg_speedup')[['operator', 'avg_speedup']].values.tolist(),
        'fast_ops': success_df[success_df['avg_speedup'] > 2.0].sort_values('avg_speedup', ascending=False)[['operator', 'avg_speedup']].values.tolist(),
        # 成功算子的详细数据用于绘图（使用 func_name）
        'success_ops_detail': df[df['include_in_stats']][['func_name', 'avg_speedup']].values.tolist(),
        'no_accuracy_ops': no_accuracy['operator'].tolist(),
        'accuracy_failed_ops': [
            (op, f, e, read_log_file(folder, op, "accuracy") if folder else "")
            for op, f, e in accuracy_failed[['operator', 'failed', 'errors']].values.tolist()
        ],
        'no_perf_ops': [
            (op, read_log_file(folder, op, "perf") if folder else "")
            for op in no_perf['operator'].tolist()
        ],
    }


def generate_html(stats, folder_name):
    """生成 HTML 报告"""
    import json
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 生成柱状图数据
    chart_data = json.dumps(stats['success_ops_detail'])

    # 计算百分比
    if stats['total_speedups'] > 0:
        pct_below = stats['below_08'] / stats['total_speedups'] * 100
        pct_between = stats['between_08_1'] / stats['total_speedups'] * 100
        pct_above = stats['above_1'] / stats['total_speedups'] * 100
    else:
        pct_below = pct_between = pct_above = 0

    # 生成需关注算子表格行
    slow_rows = ""
    for op, speedup in stats['slow_ops']:
        slow_rows += f'<tr><td>{op}</td><td><span class="badge badge-danger">{speedup:.4f}</span></td></tr>\n'

    # 生成高性能算子表格行
    fast_rows = ""
    for op, speedup in stats['fast_ops']:
        fast_rows += f'<tr><td>{op}</td><td><span class="badge badge-success">{speedup:.4f}</span></td></tr>\n'

    # 生成失败算子部分
    failed_section = ""
    if stats['no_accuracy_count'] > 0 or stats['accuracy_failed_count'] > 0 or stats['no_perf_count'] > 0:
        failed_section = f'''
        <div class="card">
            <div class="card-header">失败算子分析</div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
        '''
        if stats['no_accuracy_count'] > 0:
            ops_list = '<br>'.join(stats['no_accuracy_ops'])
            failed_section += f'''
                    <div style="background: #fff5f5; border-left: 4px solid #fc8181; padding: 20px; border-radius: 8px; max-height: 400px; overflow-y: auto;">
                        <h4 style="color: #c53030; margin-bottom: 10px;">无精度测试用例 ({stats['no_accuracy_count']}个)</h4>
                        <p style="color: #742a2a; font-size: 0.9rem;">{ops_list}</p>
                    </div>
            '''
        if stats['accuracy_failed_count'] > 0:
            import html as html_module
            ops_items = []
            for idx, (op, f, e, log) in enumerate(stats['accuracy_failed_ops']):
                escaped_log = html_module.escape(log)
                ops_items.append(f'''<span class="failed-op" onclick="showLog('log-{idx}')" style="cursor: pointer; text-decoration: underline; color: #c53030;">{op}</span> (failed={int(f)}, errors={int(e)})
                <div id="log-{idx}" data-title="{op} - accuracy.log" style="display: none;">{escaped_log}</div>''')
            ops_list = '<br>'.join(ops_items)
            failed_section += f'''
                    <div style="background: #fff5f5; border-left: 4px solid #fc8181; padding: 20px; border-radius: 8px; max-height: 600px; overflow-y: auto;">
                        <h4 style="color: #c53030; margin-bottom: 10px;">精度测试失败 ({stats['accuracy_failed_count']}个) <span style="font-size: 0.8rem; font-weight: normal;">- 点击算子名查看日志</span></h4>
                        <p style="color: #742a2a; font-size: 0.9rem;">{ops_list}</p>
                    </div>
            '''
        if stats['no_perf_count'] > 0:
            perf_items = []
            for idx, (op, log) in enumerate(stats['no_perf_ops']):
                escaped_log = html_module.escape(log)
                perf_items.append(f'''<span class="failed-op" onclick="showLog('perf-log-{idx}')" style="cursor: pointer; text-decoration: underline; color: #975a16;">{op}</span>
                <div id="perf-log-{idx}" data-title="{op} - perf.log" style="display: none;">{escaped_log}</div>''')
            ops_list = '<br>'.join(perf_items)
            failed_section += f'''
                    <div style="background: #fffff0; border-left: 4px solid #ecc94b; padding: 20px; border-radius: 8px; max-height: 600px; overflow-y: auto;">
                        <h4 style="color: #975a16; margin-bottom: 10px;">无性能测试结果 ({stats['no_perf_count']}个) <span style="font-size: 0.8rem; font-weight: normal;">- 点击算子名查看日志</span></h4>
                        <p style="color: #744210; font-size: 0.9rem;">{ops_list}</p>
                    </div>
            '''
        failed_section += '''
                </div>
            </div>
        </div>
        '''

    # 优先优化算子
    priority_ops = [f"<strong>{op}</strong> ({speedup:.2f})" for op, speedup in stats['slow_ops'][:4]]
    priority_html = "<br>".join(priority_ops) if priority_ops else "无"

    # 次优先优化算子
    secondary_ops = [f"<strong>{op}</strong> ({speedup:.2f})" for op, speedup in stats['slow_ops'][4:8]]
    secondary_html = "<br>".join(secondary_ops) if secondary_ops else "无"

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>算子测试结果分析报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; color: white; margin-bottom: 40px; }}
        .header h1 {{ font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }}
        .card {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            margin-bottom: 30px;
            overflow: hidden;
        }}
        .card-header {{
            background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
            color: white;
            padding: 20px 30px;
            font-size: 1.3rem;
            font-weight: 600;
        }}
        .card-body {{ padding: 30px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .stat-item {{
            background: linear-gradient(135deg, #f6f8fc 0%, #eef2f7 100%);
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            transition: transform 0.3s ease;
        }}
        .stat-item:hover {{ transform: translateY(-5px); }}
        .stat-value {{ font-size: 2.5rem; font-weight: 700; color: #5a67d8; margin-bottom: 8px; }}
        .stat-value.success {{ color: #38a169; }}
        .stat-label {{ color: #718096; font-size: 0.95rem; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 15px 20px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f7fafc; font-weight: 600; color: #4a5568; text-transform: uppercase; font-size: 0.85rem; }}
        tr:hover {{ background: #f7fafc; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; }}
        .badge-success {{ background: #c6f6d5; color: #22543d; }}
        .badge-warning {{ background: #feebc8; color: #744210; }}
        .badge-danger {{ background: #fed7d7; color: #742a2a; }}
        .summary-box {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }}
        .summary-item {{ text-align: center; padding: 20px; background: #f7fafc; border-radius: 10px; }}
        .summary-item .value {{ font-size: 1.8rem; font-weight: 700; color: #2d3748; }}
        .summary-item .label {{ font-size: 0.9rem; color: #718096; margin-top: 5px; }}
        .distribution-chart {{ display: flex; height: 40px; border-radius: 8px; overflow: hidden; margin: 20px 0; }}
        .dist-segment {{ display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.9rem; }}
        .dist-low {{ background: linear-gradient(90deg, #fc8181, #f56565); }}
        .dist-medium {{ background: linear-gradient(90deg, #f6ad55, #ed8936); }}
        .dist-high {{ background: linear-gradient(90deg, #68d391, #48bb78); }}
        .legend {{ display: flex; justify-content: center; gap: 30px; margin-top: 15px; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 0.9rem; color: #4a5568; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
        .env-info {{ display: flex; justify-content: center; gap: 40px; flex-wrap: wrap; }}
        .env-item {{ display: flex; align-items: center; gap: 8px; color: rgba(255,255,255,0.9); }}
        .env-item svg {{ width: 18px; height: 18px; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
        .section-title {{ font-size: 1.1rem; color: #4a5568; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #e2e8f0; }}
        .op-list {{ max-height: 400px; overflow-y: auto; }}
        .op-list::-webkit-scrollbar {{ width: 6px; }}
        .op-list::-webkit-scrollbar-track {{ background: #f1f1f1; border-radius: 3px; }}
        .op-list::-webkit-scrollbar-thumb {{ background: #c1c1c1; border-radius: 3px; }}
        .footer {{ text-align: center; color: rgba(255,255,255,0.7); margin-top: 30px; font-size: 0.9rem; }}
        @media (max-width: 768px) {{
            .two-col {{ grid-template-columns: 1fr; }}
            .summary-box {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        .failed-op:hover {{ background: #fed7d7; padding: 2px 4px; border-radius: 4px; }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.7);
        }}
        .modal-content {{
            background: #1a1a2e;
            color: #eee;
            margin: 3% auto;
            padding: 20px;
            border-radius: 12px;
            width: 90%;
            max-width: 1200px;
            height: 85vh;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            position: relative;
        }}
        .modal-close {{
            position: sticky;
            top: 0;
            float: right;
            color: #fff;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            background: #c53030;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            line-height: 32px;
            text-align: center;
        }}
        .modal-close:hover {{ background: #e53e3e; }}
        .modal-title {{
            color: #68d391;
            font-size: 1.1rem;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #4a5568;
        }}
    </style>
    <script>
        function showLog(logId) {{
            var content = document.getElementById(logId).innerHTML;
            var title = document.getElementById(logId).getAttribute('data-title') || '日志详情';
            document.getElementById('modal-title').innerText = title;
            document.getElementById('modal-body').innerHTML = content;
            document.getElementById('logModal').style.display = 'block';
        }}
        function closeModal() {{
            document.getElementById('logModal').style.display = 'none';
        }}
        window.onclick = function(event) {{
            var modal = document.getElementById('logModal');
            if (event.target == modal) {{
                modal.style.display = 'none';
            }}
        }}
        document.onkeydown = function(evt) {{
            evt = evt || window.event;
            if (evt.keyCode == 27) {{
                closeModal();
            }}
        }};
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>算子测试结果分析报告</h1>
            <div class="env-info" style="margin-top: 20px;">
                <div class="env-item">
                    <svg fill="currentColor" viewBox="0 0 20 20"><path d="M2 6a2 2 0 012-2h5l2 2h5a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z"/></svg>
                    <span>{folder_name}</span>
                </div>
                <div class="env-item">
                    <svg fill="currentColor" viewBox="0 0 20 20"><path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/><path fill-rule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clip-rule="evenodd"/></svg>
                    <span>{stats['total']} 个算子</span>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">1. 概览</div>
            <div class="card-body">
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value">{stats['total']}</div>
                        <div class="stat-label">总算子数量</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value success">{stats['success_count']}</div>
                        <div class="stat-label">精度正确且有性能结果</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value success">{stats['accuracy_failed_count']}</div>
                        <div class="stat-label">精度测试失败</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value success">{stats['no_accuracy_count']}</div>
                        <div class="stat-label">无精度测试用例</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value success">{stats['no_perf_count']}</div>
                        <div class="stat-label">无性能结果</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">2. 加速比统计</div>
            <div class="card-body">
                <div class="summary-box">
                    <div class="summary-item">
                        <div class="value">{stats['median']:.4f}</div>
                        <div class="label">中位数</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">{stats['mean']:.4f}</div>
                        <div class="label">平均值</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">{stats['min']:.4f}</div>
                        <div class="label">最小值</div>
                    </div>
                    <div class="summary-item">
                        <div class="value">{stats['max']:.4f}</div>
                        <div class="label">最大值</div>
                    </div>
                </div>

                <h3 class="section-title">加速比分布</h3>
                <div class="distribution-chart">
                    <div class="dist-segment dist-low" style="flex: {pct_below};">{pct_below:.1f}%</div>
                    <div class="dist-segment dist-medium" style="flex: {pct_between};">{pct_between:.1f}%</div>
                    <div class="dist-segment dist-high" style="flex: {pct_above};">{pct_above:.1f}%</div>
                </div>
                <div class="legend">
                    <div class="legend-item"><div class="legend-dot" style="background: #f56565;"></div><span>&lt; 0.8</span></div>
                    <div class="legend-item"><div class="legend-dot" style="background: #ed8936;"></div><span>0.8 ~ 1.0</span></div>
                    <div class="legend-item"><div class="legend-dot" style="background: #48bb78;"></div><span>&gt; 1.0</span></div>
                </div>

                <table style="margin-top: 30px;">
                    <thead>
                        <tr><th>区间</th><th>数量</th><th>占比</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><span class="badge badge-danger">&lt; 0.8</span></td><td>{stats['below_08']}</td><td>{pct_below:.2f}%</td></tr>
                        <tr><td><span class="badge badge-warning">0.8 ~ 1.0</span></td><td>{stats['between_08_1']}</td><td>{pct_between:.2f}%</td></tr>
                        <tr><td><span class="badge badge-success">&gt; 1.0</span></td><td>{stats['above_1']}</td><td>{pct_above:.2f}%</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <div class="card-header">加速比柱状图</div>
            <div class="card-body">
                <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; align-items: center;">
                    <div>
                        <label style="font-size: 0.9rem; color: #4a5568; margin-right: 8px;">筛选:</label>
                        <select id="filterRange" onchange="updateChart()" style="padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px;">
                            <option value="all">全部</option>
                            <option value="below08">&lt; 0.8</option>
                            <option value="between">0.8 ~ 1.0</option>
                            <option value="above1">&gt; 1.0</option>
                        </select>
                    </div>
                    <div>
                        <label style="font-size: 0.9rem; color: #4a5568; margin-right: 8px;">排序:</label>
                        <select id="sortOrder" onchange="updateChart()" style="padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px;">
                            <option value="name">按名称</option>
                            <option value="asc">加速比升序</option>
                            <option value="desc">加速比降序</option>
                        </select>
                    </div>
                    <div>
                        <label style="font-size: 0.9rem; color: #4a5568; margin-right: 8px;">Y轴上限:</label>
                        <input type="number" id="yAxisMax" value="3" min="1" step="0.5" onchange="updateChart()" style="padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px; width: 80px;">
                    </div>
                    <div style="flex: 1; min-width: 200px;">
                        <input type="text" id="searchBox" placeholder="搜索算子名..." oninput="updateChart()" style="padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px; width: 100%;">
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span id="chartInfo" style="font-size: 0.9rem; color: #718096;"></span>
                    <div>
                        <button onclick="prevPage()" style="padding: 6px 12px; border: 1px solid #e2e8f0; border-radius: 6px; cursor: pointer; margin-right: 5px;">&lt; 上一页</button>
                        <span id="pageInfo" style="font-size: 0.9rem; color: #4a5568; margin: 0 10px;"></span>
                        <button onclick="nextPage()" style="padding: 6px 12px; border: 1px solid #e2e8f0; border-radius: 6px; cursor: pointer;">&gt; 下一页</button>
                    </div>
                </div>
                <div style="height: 400px;">
                    <canvas id="speedupChart"></canvas>
                </div>
            </div>
        </div>

        <div class="two-col">
            <div class="card">
                <div class="card-header">3. 需关注算子（加速比 &lt; 0.8）</div>
                <div class="card-body">
                    <div class="op-list">
                        <table>
                            <thead><tr><th>算子名</th><th>加速比</th></tr></thead>
                            <tbody>{slow_rows if slow_rows else '<tr><td colspan="2" style="text-align:center;color:#718096;">无</td></tr>'}</tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">4. 高性能算子（加速比 &gt; 2.0）</div>
                <div class="card-body">
                    <div class="op-list">
                        <table>
                            <thead><tr><th>算子名</th><th>加速比</th></tr></thead>
                            <tbody>{fast_rows if fast_rows else '<tr><td colspan="2" style="text-align:center;color:#718096;">无</td></tr>'}</tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        {failed_section}

        <div class="card">
            <div class="card-header">5. 优化建议</div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                    <div style="background: #fff5f5; border-left: 4px solid #fc8181; padding: 20px; border-radius: 8px;">
                        <h4 style="color: #c53030; margin-bottom: 10px;">优先优化</h4>
                        <p style="color: #742a2a; font-size: 0.95rem;">{priority_html}</p>
                    </div>
                    <div style="background: #fffff0; border-left: 4px solid #ecc94b; padding: 20px; border-radius: 8px;">
                        <h4 style="color: #975a16; margin-bottom: 10px;">次优先优化</h4>
                        <p style="color: #744210; font-size: 0.95rem;">{secondary_html}</p>
                    </div>
                    <div style="background: #f0fff4; border-left: 4px solid #68d391; padding: 20px; border-radius: 8px;">
                        <h4 style="color: #276749; margin-bottom: 10px;">优势保持</h4>
                        <p style="color: #22543d; font-size: 0.95rem;">
                            {pct_above:.1f}% 的算子获得性能提升，其中 {len(stats['fast_ops'])} 个算子加速比超过 2 倍。
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            报告生成时间: {now}
        </div>
    </div>

    <!-- 日志弹窗 -->
    <div id="logModal" class="modal">
        <div class="modal-content">
            <span class="modal-close" onclick="closeModal()">&times;</span>
            <div class="modal-title" id="modal-title">日志详情</div>
            <div id="modal-body"></div>
        </div>
    </div>

    <script>
        // 柱状图数据和控制
        const allData = {chart_data};
        let filteredData = [];
        let currentPage = 0;
        const pageSize = 50;
        let chart = null;

        function filterAndSortData() {{
            const filterRange = document.getElementById('filterRange').value;
            const sortOrder = document.getElementById('sortOrder').value;
            const searchText = document.getElementById('searchBox').value.toLowerCase();

            // 筛选
            filteredData = allData.filter(item => {{
                const [name, speedup] = item;
                // 搜索过滤
                if (searchText && !name.toLowerCase().includes(searchText)) return false;
                // 范围过滤
                if (filterRange === 'below08' && speedup >= 0.8) return false;
                if (filterRange === 'between' && (speedup < 0.8 || speedup > 1.0)) return false;
                if (filterRange === 'above1' && speedup <= 1.0) return false;
                return true;
            }});

            // 排序
            if (sortOrder === 'asc') {{
                filteredData.sort((a, b) => a[1] - b[1]);
            }} else if (sortOrder === 'desc') {{
                filteredData.sort((a, b) => b[1] - a[1]);
            }} else {{
                filteredData.sort((a, b) => a[0].localeCompare(b[0]));
            }}
        }}

        function updateChart() {{
            filterAndSortData();
            currentPage = 0;
            renderChart();
        }}

        function renderChart() {{
            const yAxisMax = parseFloat(document.getElementById('yAxisMax').value) || 3;
            const start = currentPage * pageSize;
            const end = Math.min(start + pageSize, filteredData.length);
            const pageData = filteredData.slice(start, end);

            const labels = pageData.map(item => item[0]);
            const values = pageData.map(item => item[1]);
            const displayValues = values.map(v => Math.min(v, yAxisMax));
            const colors = values.map(v => {{
                if (v < 0.8) return 'rgba(245, 101, 101, 0.8)';
                if (v <= 1.0) return 'rgba(237, 137, 54, 0.8)';
                return 'rgba(72, 187, 120, 0.8)';
            }});

            // 更新信息
            document.getElementById('chartInfo').innerText = `共 ${{filteredData.length}} 个算子，当前显示 ${{start + 1}}-${{end}}`;
            document.getElementById('pageInfo').innerText = `${{currentPage + 1}} / ${{Math.ceil(filteredData.length / pageSize) || 1}}`;

            // 销毁旧图表
            if (chart) chart.destroy();

            // 创建新图表
            const ctx = document.getElementById('speedupChart').getContext('2d');
            chart = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: '加速比',
                        data: displayValues,
                        backgroundColor: colors,
                        borderColor: colors.map(c => c.replace('0.8', '1')),
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    const realValue = values[context.dataIndex];
                                    if (realValue > yAxisMax) {{
                                        return `加速比: ${{realValue.toFixed(4)}} (截断显示)`;
                                    }}
                                    return `加速比: ${{realValue.toFixed(4)}}`;
                                }}
                            }}
                        }},
                        legend: {{ display: false }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: yAxisMax,
                            title: {{ display: true, text: '加速比' }}
                        }},
                        x: {{
                            ticks: {{
                                maxRotation: 45,
                                minRotation: 45,
                                font: {{ size: 10 }}
                            }}
                        }}
                    }}
                }}
            }});
        }}

        function prevPage() {{
            if (currentPage > 0) {{
                currentPage--;
                renderChart();
            }}
        }}

        function nextPage() {{
            if ((currentPage + 1) * pageSize < filteredData.length) {{
                currentPage++;
                renderChart();
            }}
        }}

        // 初始化图表
        document.addEventListener('DOMContentLoaded', updateChart);
    </script>
</body>
</html>'''
    return html


def main():
    folder = "D:/Users/Downloads/results_20260303_015418"
    if len(sys.argv) < 2:
        print("用法: python analyze_ops.py <结果文件夹路径>")
        print("示例: python analyze_ops.py /path/to/results_folder")
        sys.exit(1)

    folder = Path(sys.argv[1])
    print(folder)

    if not folder.exists():
        print(f"错误: 文件夹不存在: {folder}")
        sys.exit(1)

    # 查找 summary.xlsx
    xlsx_path = folder / "summary.xlsx"
    if not xlsx_path.exists():
        xlsx_path = folder / "summary.excel"
    if not xlsx_path.exists():
        print(f"错误: 未找到 summary.xlsx 或 summary.excel")
        sys.exit(1)

    print(f"读取文件: {xlsx_path}")

    # 读取数据
    df = pd.read_excel(xlsx_path)

    # 统计有 operator 名称的行数（算子数量）
    op_count = df['operator'].notna().sum()
    print(f"共 {op_count} 个算子，{len(df)} 行数据")

    # 分析数据
    stats = analyze_data(df, folder)

    # 生成 HTML
    html = generate_html(stats, folder.name)

    # 保存报告
    output_path = folder / "report.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n报告已生成: {output_path}")

    # 打印摘要
    print(f"\n===== 分析摘要 =====")
    print(f"总算子数: {stats['total']}")
    print(f"成功数: {stats['success_count']}")
    print(f"加速比中位数: {stats['median']:.4f}")
    print(f"加速比平均值: {stats['mean']:.4f}")
    print(f"加速比 < 0.8: {stats['below_08']} 个")
    print(f"加速比 > 1.0: {stats['above_1']} 个")


if __name__ == "__main__":
    main()
