Errors
======

This page lists a few of the errors you might encounter when using imlearn, along with representative examples of
how one might fix them.

Using engine = "matlab"
-----------------------

In order to use matlab as an engine, you will need to have Octave (MATLAB) in your machine. In linux, you can install
it using the following commands:

.. code:: bash

    sudo apt install octave

For other platforms, please refer to the official guides: https://octave.org/download

Missing packages
^^^^^^^^^^^^^^^^

When using matlab as engine, some algorithms (such as DAIMC and OSLFIMVC) could have some extra dependencies. To install
these dependencies, try the following commands in a terminal:

.. code:: bash

    sudo apt install octave-control
    sudo apt install octave-statistics

Using engine = "r"
-----------------------

In order to use R as an engine, you will need to have R in your machine. In linux, you can install it using the
following commands:

.. code:: bash

    sudo apt install r-base r-base-dev -y

For other platforms, please refer to the official guides: https://cran.r-project.org/doc/manuals/r-patched/R-admin.html

Missing packages
^^^^^^^^^^^^^^^^

When using R as engine, some algorithms (such as jNMF) could have some extra dependencies. To install these
dependencies, try the following commands in R:

.. code:: R

    install.packages("nnTensor")


Compatibility with other packages
---------------------------------

Currently, we do not support the following:

* Numpy>=2: There is an incompatibility with Scipy that prevents support for this version. For details, see
https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility

* SciPy>=1.13. The scipy.linalg.tri function has been removed in favor of numpy.tri. Unfortunately, the gemsim package has not yet been updated to support this change.



