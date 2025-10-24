Algorithm selection guide
==============================

This page provides a quick overview of the available algorithms in `iMML` and the
supported input modalities. Use this table to choose an appropriate method for your
task and check whether additional modules or dependencies are required.

.. list-table::
   :header-rows: 1
   :widths: 10 5 70 5 10
   :align: center

   * - Task
     - Algorithm
     - Input modalities
     - Module
     - Extra dependencies
   * - Classification
     - :class:`~imml.classify.M3Care`
     - Numeric | Image | Text
     - deep
     -
   * - Classification
     - :class:`~imml.classify.MUSE`
     - Numeric | Text | Time series
     - deep
     -
   * - Classification
     - :class:`~imml.classify.RAGPT`
     - Image | Text
     - deep
     -
   * - Clustering
     - :class:`~imml.clustering.DAIMC`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.clustering.EEIMVC`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.clustering.IMSCAGL`
     - Numeric
     - matlab
     - octave
   * - Clustering
     - :class:`~imml.clustering.IMSR`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.clustering.IntegrAO`
     - Numeric
     - deep
     -
   * - Clustering
     - :class:`~imml.clustering.LFIMVC`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.clustering.MKKMIK`
     - Numeric
     - matlab
     - octave
   * - Clustering
     - :class:`~imml.clustering.MONET`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.clustering.MRGCN`
     - Numeric
     - deep
     -
   * - Clustering
     - :class:`~imml.clustering.NEMO`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.clustering.OMVC`
     - Numeric
     - matlab
     - octave
   * - Clustering
     - :class:`~imml.clustering.OPIMC`
     - Numeric
     - matlab
     - octave
   * - Clustering
     - :class:`~imml.clustering.OSLFIMVC`
     - Numeric
     - matlab
     - octave, octave-statistics
   * - Clustering
     - :class:`~imml.clustering.PIMVC`
     - Numeric
     - matlab
     - octave
   * - Clustering
     - :class:`~imml.clustering.SIMCADC`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.clustering.SUMO`
     - Numeric
     -
     -
   * - Decomposition
     - :class:`~imml.decomposition.DFMF`
     - Numeric
     -
     -
   * - Decomposition
     - :class:`~imml.decomposition.MOFA`
     - Numeric
     -
     -
   * - Decomposition
     - :class:`~imml.decomposition.JNMF`
     - Numeric
     - r
     - R, nnTensor
   * - Feature selection
     - :class:`~imml.feature_selection.JNMFFeatureSelection`
     - Numeric
     - r
     - R, nnTensor
   * - Impute
     - :class:`~imml.impute.DFMFImputer`
     - Numeric
     -
     -
   * - Impute
     - :class:`~imml.impute.MOFAImputer`
     - Numeric
     -
     -
   * - Impute
     - :class:`~imml.impute.JNMFImputer`
     - Numeric
     - r
     - R, nnTensor
   * - Retrieve
     - :class:`~imml.retrieve.MCR`
     - Image | Text
     - deep
     -
   * - Statistics
     - :class:`~imml.statistics.pid`
     - Numeric
     -
     -

How to install an additional module
----------------------------------------------------------

See our `page <https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies>`__ on
how to install a module.

How to install extra dependencies
----------------------------------------------------------

Extra dependencies when using "matlab" module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use 'matlab' as an engine, you will need to have `Octave` (`MATLAB`) in your machine. In linux, you can
install it using the following commands:

.. code:: bash

    sudo apt install octave

For other platforms, please refer to the `official guides <https://octave.org/download>`__.

Additionally, to install extra dependencies, execute the following commands in a terminal:

.. code:: bash

    sudo apt install octave-control
    sudo apt install octave-statistics

Extra dependencies when using "r" module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use 'r' as an engine, you will need to have R in your machine. In linux, you can install it using the
following commands:

.. code:: bash

    sudo apt install r-base r-base-dev -y

For other platforms, please refer to the `official guides <https://cran.r-project.org/doc/manuals/r-patched/R-admin.html>`__.

Additionally, to install extra dependencies, execute the following commands in R:

.. code:: R

    install.packages("nnTensor")
