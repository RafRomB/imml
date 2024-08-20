imlearn: A Python package for incomplete multi-view learning
=============================================================

Overview
--------

This package is designed for **incomplete multi-view learning (IML)**, a field in
machine learning where data that is represented by multiple views or 
modalities where some of them are missing for some of the samples.
It implements different algorithms, transformers, utils, and datasets for
this task.

IML is usually a hard problem because of the following reasons:

- Heterogeneity of views: IML deals with data represented by multiple views, each of which may have a different representation or encoding of the same underlying information. This heterogeneity can make it challenging to compare and integrate the different views, especially when some of the views are missing.
- Missing data: In IML, some views may be missing for some samples, making it even more challenging. This missing data can lead to incomplete or biased representations of the underlying patterns in the data.
- Dimensionality: IML often involves high-dimensional data, which can make difficult to extract meaningful patterns in the data. Moreover, different views may have different dimensions, which can further complicate the learning process.
- Scalability: IML may involve large amounts of data, making it computationally expensive and challenging to scale to larger datasets.

Addressing these challenges requires the development of specialized algorithms and techniques that can effectively
handle incomplete multi-view data and extract meaningful patterns from it.

Usage
-----

This package provides a user-friendly interface to apply these algorithms to
user-provided data. Moreover, it is compatible with Scikit-learn and can be easily
integrated into Scikit-learn pipelines for data preprocessing and modeling. So same
than it, classes implement the fit method that creates a model by taking as input
the data, and predict method, that returns the predictions for each sample.

We show a simple example of how it works.

.. code:: python

    # Import an algorithm, in this case, for clustering
    from imlearn.cluster import DAIMC
    # Load an (incomplete) multi-view dataset.
    from imlearn.datasets import LoadDataset
    Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    # Create an instance of an algorithm
    estimator = DAIMC(n_clusters = 3, random_state=42)
    # Fit the model and get predictions
    labels = estimator.fit_predict(Xs)


Citation
--------

If you use this package in your project, please consider citing the 
following paper:

[INSERT CITATION HERE]

In addition, we kindly request that you cite the package itself:

[INSERT BIBTEX OR CITATION HERE]

Thank you for acknowledging the use of this package in your research!
