# Incomplete Multi-View Clustering

This package is designed for Incomplete Multi-View Clustering, which is a 
technique used to cluster data that is represented by multiple views or 
modalities where some of the views are missing for some of the samples.
It implements different transformers, algorithms and pipelines for 
Incomplete Multi-View Clustering.

Incomplete multi-view clustering is usually a hard problem because of the following reasons:
- **Heterogeneity of views**: Incomplete multi-view clustering deals with data represented by multiple views, each of which may have a different representation or encoding of the same underlying information. This heterogeneity can make it challenging to compare and integrate the different views, especially when some of the views are missing. 
- **Missing data**: In incomplete multi-view clustering, some views may be missing for some samples, making it challenging to determine how to cluster the data. This missing data can lead to incomplete or biased representations of the underlying patterns in the data. 
- **Dimensionality**: Incomplete multi-view clustering often involves high-dimensional data, which can make it difficult to extract meaningful patterns and identify clusters in the data. Moreover, different views may have different dimensions, which can further complicate the clustering process. 
- **Scalability**: Incomplete multi-view clustering may involve large amounts of data, making it computationally expensive and challenging to scale to larger datasets.

Addressing these challenges requires the development of specialized algorithms and techniques that can effectively handle incomplete multi-view data and extract meaningful patterns and clusters from it.

## Installation

The easiest way to install this package is using pip. Simply run the following command:

```bash
pip install imvc
```

## Usage

This package provides a user-friendly interface to apply these algorithms to 
user-provided data. Moreover, it is compatible with Scikit-learn and can be easily 
integrated into Scikit-learn pipelines for data preprocessing and modeling. So same 
than it, classes implement the fit method that creates a model by taking as input 
the data and the missing views, and predict method, that returns the cluster labels 
for each sample.

We show a simple example of how it works.

```python
from imvc.datasets import LoadDataset
from imvc.pipelines import NMFCPipeline

# Load an incomplete multi-view dataset
Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)

# Create an instance of a pipeline using a Non-negative matrix factorization with 3 clusters
pipeline = NMFCPipeline(n_clusters = 3)

# Fit the model with the data and missing views
pipeline.fit(Xs)

# Get predictions
pipeline.predict(Xs)
```

We also provide some Jupyter notebooks in the examples/ directory to help you get 
started with using this package. These notebooks demonstrate how to use the package 
for clustering datasets and provide step-by-step instructions for each example. 
For more details on the usage and available options for each class, please refer 
to the documentation.

## Contributing

This package is open-source and contributions are welcome! If you find any issues or 
have suggestions for improvement, please create an issue or pull request on the 
Github repository.

## Citation

If you use this package in your research, please consider citing the 
following paper:

[INSERT CITATION HERE]

In addition, we kindly request that you cite the package itself:

[INSERT BIBTEX OR CITATION HERE]

Thank you for acknowledging the use of this package in your research!