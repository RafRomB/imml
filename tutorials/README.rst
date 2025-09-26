Tutorials
====================

This folder collects hands‑on, runnable examples that demonstrate how to use `iMML` for common multi‑modal learning
workflows, including exploring datasets, simulating missing modalities, and clustering with incomplete data,
among others. Each script is self‑contained and designed to be easy to adapt to your own data.

Where to read them
-------------------------
- Documentation: https://imml.readthedocs.io/en/latest/auto_tutorials/index.html.
  The online gallery renders these scripts as rich, formatted pages with figures.
- In-repo: https://github.com/ocbe-uio/imml/tree/main/tutorials. Tutorials scripts are hosted here.


Prerequisites
-------------
To run these tutorials and generate the figures, we recommend installing the optional tutorial dependencies:

  pip install imml[tutorials]

You can, of course, run the scripts with a standard `imml` installation if you already have the required packages.

How to run locally
------------------
- As Python scripts: run any file directly, for example:

  python tutorials/cluster_incomplete_mmd.py

  Most scripts will print intermediate results and pop up figures.
- In a notebook: copy/paste sections into Jupyter and execute them step by step.

Questions or feedback?
----------------------
- Open an issue: https://github.com/ocbe-uio/imml/issues
- Contributions are welcome: see CONTRIBUTING.md or the online contributing guide.