===============
Developer Guide
===============

Thank you for your interest in contributing to ConfUSIus! This document provides
guidelines and instructions for contributing to the project.

Development Setup
=================

1. Clone the repository
***********************

.. code-block:: bash

    git clone https://github.com/sdiebolt/confusius.git
    cd confusius

2. Install dependencies
***********************

We use `uv <https://docs.astral.sh/uv/>`_ for dependency management. Install all
dependencies including development tools:

.. code-block:: bash

    uv sync

3. Install pre-commit hooks
****************************

We use pre-commit hooks to ensure code quality:

.. code-block:: bash

    uvx pre-commit install

Code Style
==========

ConfUSIus follows strict code style guidelines to ensure consistency and quality:

Formatting
**********

- Code is formatted with `Ruff <https://github.com/astral-sh/ruff>`_.

Type Hints
**********

- Use comprehensive type hints with ``numpy.typing`` for arrays.
- Use ``Literal`` for string literal types.
- Use ``TypedDict`` for structured data dictionaries.

Documentation
*************

- Use `numpydoc <https://numpydoc.readthedocs.io/en/latest/>`_ format for all public
  functions.
- Include ``Parameters``, ``Returns``, and ``Raises`` sections.
- Document complex algorithms with references.
- Include default values in parameter documentation as ``arg : type, default: value``.
- For optional parameters with ``None`` default, use ``arg : type, optional``.

Naming Conventions
******************

- Functions/methods: ``snake_case``.
- Classes: ``PascalCase``.
- Constants: ``UPPER_CASE``.
- Private functions/methods: leading underscore ``_function_name``.

Running Quality Checks
=======================

Linting and formatting
**********************

.. code-block:: bash

    # Check code with Ruff
    uvx ruff check .

    # Format code with Ruff
    uvx ruff format .

    # Run all pre-commit hooks
    uvx pre-commit run --all-files

Type checking
*************

.. code-block:: bash

    uvx ty check

Testing
*******

.. code-block:: bash

    # Run all tests
    uv run pytest

    # Run tests with verbose output
    uv run pytest -v

    # Run specific test file
    uv run pytest src/confusius/io/tests/test_autc.py

    # Run tests excluding slow tests
    uv run pytest -m "not slow"

    # Run tests excluding those requiring real data
    uv run pytest -m "not real_data"

Documentation
*************

.. code-block:: bash

    # Build documentation
    uv run sphinx-build -j auto docs/ docs/_build/html

    # Clean documentation build
    rm -rf docs/_build/

Writing Tests
=============

Test Organization
*****************

Tests are organized in a modular structure:

- ``src/confusius/tests/`` - Package-level tests.
- ``src/confusius/io/tests/`` - Tests for I/O module.
- ``src/confusius/iq/tests/`` - Tests for IQ processing module.

Each test directory contains:

- ``__init__.py`` - Makes directory a Python package.
- ``conftest.py`` - Shared fixtures for tests.
- ``test_*.py`` - Test files.

Test Fixtures
*************

Use pytest fixtures defined in ``conftest.py`` files for reusable test data:

.. code-block:: python

    def test_my_function(synthetic_iq_small):
        """Test with small synthetic IQ data."""
        result = my_function(synthetic_iq_small)
        assert result is not None

Test Markers
************

Use appropriate markers for tests:

- ``@pytest.mark.slow`` - For tests that take significant time.
- ``@pytest.mark.real_data`` - For tests requiring real data files.

.. code-block:: python

    @pytest.mark.real_data
    def test_load_real_file(autc_sample_files):
        """Test loading real AUTC files."""
        # Test implementation

Writing Good Tests
******************

- Test one thing at a time
- Use descriptive test names
- Include docstrings explaining what is being tested
- Use assertions with clear error messages
- Test edge cases and error conditions
- Use synthetic data when possible for reproducibility

Pull Request Process
=====================

1. Create a new branch for your feature or bugfix:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. Make your changes following the code style guidelines.

3. Add tests for new functionality.

4. Run all quality checks:

   .. code-block:: bash

       uvx ruff check .
       uvx ty check
       uv run pytest

5. Commit your changes with clear, descriptive commit messages.

6. Push your branch and create a pull request on GitHub.

7. Ensure all CI checks pass.

Reporting Issues
================

When reporting issues, please include:

- A clear description of the problem.
- Steps to reproduce the issue.
- Expected behavior vs. actual behavior.
- Your environment (Python version, OS, etc.).
- Relevant code snippets or error messages.

Questions
=========

If you have questions about contributing, please open an issue on GitHub or
contact the maintainers.

