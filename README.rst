|version| |python| |ruff| |ty| |uv| |codecov|

.. |version| image:: https://img.shields.io/badge/version-0.0.1-orange.svg
    :target: https://github.com/sdiebolt/confusius/
    :alt: ConfUSIus version

.. |python| image:: https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg
    :target: https://www.python.org/
    :alt: Python

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. |ty| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json
    :target: https://github.com/astral-sh/ty
    :alt: Ty

.. |uv| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
    :target: https://github.com/astral-sh/uv
    :alt: uv

.. |codecov| image:: https://codecov.io/github/sdiebolt/confusius/graph/badge.svg?token=JX2R08XZ3W
    :target: https://codecov.io/github/sdiebolt/confusius
    :alt: Codecov

ConfUSIus
=========

*ConfUSIus is currently in pre-alpha and under active development. The API is subject to change, and features may be incomplete or unstable.*

.. start-about

ConfUSIus is a Python package for handling, visualization, preprocessing, and
statistical analysis of functional ultrasound imaging (fUSI) data.

.. stop-about

Installation
------------

.. start-installation

1. Setup a virtual environment
******************************

We recommend that you install `ConfUSIus` in a virtual environment to avoid
dependency conflicts with other Python packages. Using `uv
<https://docs.astral.sh/uv/guides/install-python/>`_, you may create a new
project folder with a virtual environment as follows:

.. code-block:: bash

    uv init new_project

If you already have a project folder, you may create a virtual environment as follows:

.. code-block:: bash

    uv venv

2. Install ConfUSIus from source
********************************

ConfUSIus is a package developed by Samuel Le Meur-Diebolt at the Cortexlab and is not 
yet available on PyPI. You can add it as a project dependency directly from the GitHub
repository using the following command if your project is managed with `uv`:

.. code-block:: bash

    uv add git+https://github.com/sdiebolt/confusius.git

Or if you simply created a virtual environment using `uv venv`:

.. code-block:: bash

    source .venv/bin/activate
    uv pip install git+https://github.com/sdiebolt/confusius.git

3. Check installation
*********************

Check that ConfUSIus is correctly installed by opening a Python interpreter and
importing the package:

.. code-block:: python

  import confusius

If no error is raised, you have installed ConfUSIus correctly.

.. stop-installation
