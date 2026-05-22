%global debug_package %{nil}

# Filter out auto-generated Requires for ML/non-distro runtime deps
# (torch GPU build, triton, hydra-core...): the distro has no GPU-capable
# python3-torch, no current triton, no hydra-core; users install matching
# versions via pip per packaging/INSTALL.md. Match python3dist and
# python3.X.Ydist forms (Fedora >= 39 uses the latter).
%global __requires_exclude ^python3.*dist.*(torch)

Name:           python3-flag-audio
Version:        0.1.0
Release:        1%{?dist}
Summary:        FlagAudio — audio processing kernels for FlagOS

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagAudio
Source0:        %{url}/archive/v%{version}/flag-audio-%{version}.tar.gz
BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools >= 60
BuildRequires:  python3-wheel
BuildRequires:  python3-pip
BuildRequires:  pyproject-rpm-macros

%description
Audio-domain operators (mel spectrogram, STFT/iSTFT, resampling) accelerated via Triton on FlagOS hardware.

%prep
%autosetup -n flag-audio-%{version}

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files flag_audio

%check
# Smoke find_spec test (no actual import) — verifies the built module
# lands at the expected sitelib path. Doesn't import the module so
# missing runtime deps (torch, triton, ...) don't trip the check;
# those are user-install-time concerns, not packaging concerns.
PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=%{buildroot}%{python3_sitelib} \
    python3 -c "import importlib.util; s = importlib.util.find_spec('flag_audio'); assert s and s.origin, 'flag_audio not findable'; print('OK: flag_audio at', s.origin)"

%files -f %{pyproject_files}
%license LICENSE*

%changelog
* Wed May 13 2026 FlagOS Contributors <contact@flagos.io> - 0.1.0-1
- Initial RPM packaging.
