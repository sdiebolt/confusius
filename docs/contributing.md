---
hide:
    - navigation
---

# Contributing

We welcome contributions to ConfUSIus! This guide will help you get started.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Make your changes
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/confusius.git
cd confusius

# Install with development dependencies
uv sync

# Run tests
just test

# Run pre-commit hooks
just pre-commit
```

## Code Style

- Follow the existing code style
- Run `just pre-commit` before committing
- Add tests for new functionality
- Update documentation as needed

## Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists
2. Create a new issue with a clear description
3. Include steps to reproduce (for bugs)
4. Include code examples if applicable

## Code of Conduct

Be respectful and constructive in all interactions.
