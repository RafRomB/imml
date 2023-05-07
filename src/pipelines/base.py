from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


class BasePipeline(Pipeline):
    
    def __init__(self, estimator = KMeans(), transformers = [], verbose = False, **args):
        self.estimator = estimator
        self.transformers = transformers
        self.verbose = verbose
        super().__init__(make_pipeline(*transformers, estimator).set_params(**args).steps, verbose = verbose)
