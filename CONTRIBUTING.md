# Contributing to PaperMind AI

Thanks for considering a contribution. Bug reports, focused fixes, tests, and
well-scoped feature proposals are all useful. For larger changes, open an issue
first so we can agree on the problem and approach.

## Set up locally

PaperMind requires Python 3.12 or newer. From the repository root, create and
activate a virtual environment, then install the project and its runtime and
development dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux (use this instead of the Windows command)
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the dashboard with `streamlit run src/papermind/ui/app.py`. The embedding
model is downloaded by Sentence Transformers the first time it is needed.
Question answering can use the built-in offline synthesizer; external provider
credentials are only needed when selecting a provider that requires them.

## Verify changes

Run the relevant tests while developing, then the same checks used in CI before
opening a pull request:

```bash
pytest --cov=src/papermind tests/
ruff check src tests
black --check src tests
```

If running tests from a source checkout, set `PYTHONPATH=src` first if the
package is not otherwise importable (`$env:PYTHONPATH = "src"` in PowerShell;
`export PYTHONPATH=src` in macOS/Linux shells).

Keep changes focused, add or update tests for behavior changes, and update the
README or user guide when usage changes. Never commit API keys, private papers,
generated model files, or local cache contents.

## Open a pull request

1. Create a branch from `main` for one logical change.
2. Describe the user problem, the solution, and how you verified it.
3. Link the related issue, if there is one, and include screenshots for UI
   changes where useful.
4. Be open to review feedback; maintainers may ask for a smaller or revised
   change before merging.

For bug reports, include reproducible steps, expected and actual behavior, and
relevant error output. Remove secrets and private document content first.
