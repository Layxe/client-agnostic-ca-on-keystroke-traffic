# Client–Agnostic Continuous Authentication via Keystroke–Induced Traffic Patterns

## DOIs

**Snapshot v1.0.0** (Initial release)

[![DOI](https://zenodo.org/badge/994959236.svg)](https://doi.org/10.5281/zenodo.15579139)

## Folder Structure

```
- data/                    # Contains the keystroke websocket logs
    - yyyymmdd-hhmmss.log  # Log file with websocket packet data for a specific user
    - ...
- lib/                     # Library scripts used for preprocessing and data handling
- Baseline-Model.ipynb     # Jupyter Notebook for baseline model implementation
- Siamese-Network.ipynb    # Jupyter Notebook for Siamese Network implementation
```

## Installation

For installation, please use the package manager [uv](https://docs.astral.sh/uv/). After installing uv, run the following command in the root directory of the repository:

```bash
uv run install
```

## Usage

Run the jupyter notebooks using the created `.venv` environment.
