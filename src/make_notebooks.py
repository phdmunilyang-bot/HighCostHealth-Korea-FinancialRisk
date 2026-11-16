"""Generate single-cell Jupyter notebooks from code strings and (optionally) execute them.
Convention: each notebook contains exactly ONE code cell whose first line is an English
comment with the notebook name. Run: python src/make_notebooks.py <which>
"""
import sys, nbformat as nbf
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks"
NB.mkdir(exist_ok=True)

def write_nb(name: str, code: str):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_code_cell(code.strip("\n"))]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    (NB / name).write_text(nbf.writes(nb), encoding="utf-8")
    print("wrote", name)
