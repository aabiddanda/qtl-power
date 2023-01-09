Testing
=======

This python package is supported by substantial unit test coverage. In particular we favor `property-based testing <https://medium.com/criteo-engineering/introduction-to-property-based-testing-f5236229d237>`_ using the `hypothesis <https://hypothesis.readthedocs.io/en/latest/index.html>`_ package. Property-based testing is mainly used to ensure that across a wide variety of inputs consistent with numerical boundaries for the arguments for a function, that the function arguments are still well supported (e.g. projected power is still within the range of 0.0 - 1.0).

In order to run the unit tests (and ensure changes have not broken previous testing)::

  pip install pytest-cov pytest-xdist hypothesis
  pytest -n auto --cov

this will ensure you apply all of the unit tests using multiple available cores on your machine and a coverage report will be generated. Note that if you have setup the conda environment using the `environment.yaml` file you will not have to run the `pip install` commands.
