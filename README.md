# README

To get started with development of code in this repo, run the following from a bash shell:

- Create a virtual python environment via either `python -m venv env` or `conda create -n NAME python=VERSION`
- Activate the virtual environment via `source env/bin/activate` or `conda activate NAME`
- Install the pre-commit package manager and its dependencies into the virtual environment via `pip install .[dev]`
- Install the git hooks scripts for pre-commit into .git/hooks via `pre-commit install`
- Run pre-commit on all files via `pre-commit run --all-files`

When running `pre-commit` for the first time in the final step,
the tools used by `pre-commit` (e.g. `ruff` and `black`, as defined in `.pre-commit-config.yaml`)
will be installed into a virtual environment within `~/.cache/pre-commit`.

Individual `pre-commit` hooks can be run via `pre-commit run <hook-id>`.

Optional: if you want to be able to run tools like `black` and `ruff` outside of the `pre-commit` environment,
they need to be installed separately within the repo's virtual environment.
You can do this using `pip install .[dev-extra]`.
