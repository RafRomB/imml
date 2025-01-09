Installation
============

`iMML` can be installed using `pip` or directly from its source on GitHub. Follow the instructions below to
install `iMML` on your system.

Some features of `iMML` rely on optional dependencies. To enable these additional features, ensure you install the
required packages as described in the :ref:`optional_dependencies` section below.

.. _pipAnchor:

Instructions
-----------------------------

Using pip
^^^^^^^^^^^^

To install `iMML` using ``pip``, ensure Python 3 and ``pip`` (Python's package manager) are properly set up on
your system. If ``pip`` is not already installed or needs an update, refer to the official documentation at
https://pip.pypa.io.

Run the following command to install the most recent release of `iMML`:

.. code:: bash

    pip3 install imml

To upgrade to a newer version of `iMML`, use the `--upgrade` flag:

.. code:: bash

    pip3 install --upgrade imml

If you do not have permission, install `iMML` in your user directory by adding the `--user` flag:

.. code:: bash

    pip3 install --user imml

Manually
^^^^^^^^^^^^

Alternatively, you can download `iMML` directly from its source repository on GitHub. This method is useful if you
want to install the latest development version:

1. Clone the repository:

.. code:: bash

    git clone

2. Navigate to the project directory:

.. code:: bash

    cd imml

Install the package in editable mode:

.. code:: bash

    pip3 install -e .

This will install `iMML` and the required dependencies (see below).

Dependencies
^^^^^^^^^^^^^^^^^^^^^

`iMML` requires the following packages:

-  scikit-learn>=1.2.0
-  pandas>=1.3.3
-  numpy>=1.21.3,<2
-  scipy>=1,<1.13
-  networkx>=2.5
-  gensim>=4.2
-  h5py>=3.6
-  snfpy>=0
-  control>=0

Currently, we do not support the following:

-  Numpy>=2: There is an incompatibility with `Scipy` that prevents support for this version. For details, see this
   `thread <https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility>`_.

-  SciPy>=1.13. The ``scipy.linalg.tri`` function has been removed in favor of ``numpy.tri``. Unfortunately,
   the gemsim package has not yet been updated to support this change.

`iMML` is supported for Python >=3.8.

.. _optional_dependencies:

Optional dependencies
-----------------------------

`iMML` supports additional features that require optional dependencies. You can install these dependencies by
specifying extras during installation. For example:

-  [matlab]: Some algorithms were originally developed in Matlab. If you want to use the original implementation, use this module.
-  [r]: Some algorithms were originally developed in R. If you want to use the original implementation, use this module.
-  [torch]: Deep learning methods are included in this module.

To include this dependencies, execute in the terminal:

.. code:: bash

    pip3 install imml[keyword]

where 'keyword' is from the list above.

To install all possible dependencies:

.. code:: bash

    pip3 install imml[all]


Using engine = "matlab"
^^^^^^^^^^^^^^^^^^^^^^^

To include this dependencies, execute in the terminal:

.. code:: bash

    pip3 install imml[matlab]

In order to use matlab as an engine, you will need to have `Octave` (`MATLAB`) in your machine. In linux, you can
install it using the following commands:

.. code:: bash

    sudo apt install octave

For other platforms, please refer to the official guides: https://octave.org/download

Additionally, some algorithms (such as DAIMC and OSLFIMVC) could have some extra dependencies. To install these
dependencies, try the following commands in a terminal:

.. code:: bash

    sudo apt install octave-control
    sudo apt install octave-statistics

Using engine = "r"
^^^^^^^^^^^^^^^^^^^^^^^

To include this dependencies, execute in the terminal:

.. code:: bash

    pip3 install imml[r]

In order to use R as an engine, you will need to have R in your machine. In linux, you can install it using the
following commands:

.. code:: bash

    sudo apt install r-base r-base-dev -y

For other platforms, please refer to the official guides: https://cran.r-project.org/doc/manuals/r-patched/R-admin.html

When using R as engine, some algorithms (such as jNMF) could have some extra dependencies. To install these
dependencies, try the following commands in R:

.. code:: R

    install.packages("nnTensor")

OS Requirements
---------------
This package is supported for Linux, macOS and Windows machines.

Testing
-------

To test the package, install the testing dependencies using:

.. code:: bash

    pip3 install imml[tests]

This will install pytest and pytest-cov.

Documenting
------------

To include new documentatiob, install the documenting dependencies using:

.. code:: bash

    pip3 install imml[docs]
