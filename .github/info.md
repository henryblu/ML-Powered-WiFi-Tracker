# GitHub Configuration

Continuous integration for this project is defined under `.github/workflows`.

## File overview

- `workflows/build.yml` - builds the firmware and unit tests using the ESP-IDF
  Docker image whenever code is pushed to `main` or a pull request is opened.
- `workflows/ci.yml` - installs the Python dependencies, runs `ruff` for linting
  and executes the `pytest` suite.

## Usage

The workflows are triggered automatically when code is pushed to `main` or when
a pull request is opened. You can also trigger them manually from the GitHub
**Actions** tab. To reproduce the checks locally install the developer
dependencies and run the same tools:

```bash
python -m pip install -r dev-requirements.txt
ruff check .
pytest -q
```
