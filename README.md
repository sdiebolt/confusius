![PyPI version](https://img.shields.io/pypi/v/confusius)
![Python versions](https://img.shields.io/pypi/pyversions/confusius)
![License](https://img.shields.io/pypi/l/confusius)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![codecov](https://codecov.io/github/sdiebolt/confusius/graph/badge.svg?token=TL5AIRNSHS)](https://codecov.io/github/sdiebolt/confusius)

# ConfUSIus

> [!WARNING]
> ConfUSIus is currently in pre-alpha and under active development. The API is subject
> to change, and features may be incomplete or unstable.

ConfUSIus is a Python package for handling, visualization, preprocessing, and
statistical analysis of functional ultrasound imaging (fUSI) data.

## Installation

### 1. Setup a virtual environment

We recommend that you install ConfUSIus in a virtual environment to avoid dependency
conflicts with other Python packages. Using
[uv](https://docs.astral.sh/uv/guides/install-python/), you may create a new project
folder with a virtual environment as follows:

```bash
uv init new_project
```

If you already have a project folder, you may create a virtual environment as follows:

```bash
uv venv
```

### 2. Install ConfUSIus

ConfUSIus is available on PyPI. Install it using:

```bash
uv add confusius
```

Or with pip:

```bash
pip install confusius
```

To install the latest development version from GitHub:

```bash
uv add git+https://github.com/sdiebolt/confusius.git
```

### 3. Check installation

Check that ConfUSIus is correctly installed by opening a Python interpreter and
importing the package:

```python
import confusius
```

If no error is raised, you have installed ConfUSIus correctly.
