# Building Documentation

To build the documentation:

1. Install [ASDL](https://github.com/kazukiosawa/asdl/tree/dev-grad-maker#installation) 
2. Install [Sphinx](https://www.sphinx-doc.org/en/master/) theme via
   ```
   pip install git+https://github.com/pyg-team/pyg_sphinx_theme.git
   ```
3. Generate the documentation file via:
   ```
   cd docs
   make html
   ```

The documentation is now available to view by opening `docs/build/html/index.html`.
