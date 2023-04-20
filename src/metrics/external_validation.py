from sklearn.model_selection import ShuffleSplit


class ExternalStabilityValidation():
    
    
    def __init__(self, X, estimator = None, n_times : int = 100, sample_frac : float = 0.7, random_state : int = None):
        self.estimator = estimator
        self.sample_frac = sample_frac
        self.X = X
        self.n_times = n_times
        self.random_state = random_state
        
        train_labels = []
        val_labels = []
        
        for train_index, val_index in ShuffleSplit(n_splits=self.n_times, test_size = sample_frac, random_state = random_state).split(X):
            train_estimator = make_pipeline(FunctionTransformer(lambda x: [view.loc[train_index] for view in x]), *estimator)
            val_estimator = make_pipeline(FunctionTransformer(lambda x: [view.loc[val_index] for view in x]), *estimator)
            train_labels.append(pd.Series(train_estimator.fit_predict(X), index = train_index))
            val_labels.append(pd.Series(val_estimator.fit_predict(X), index = val_index))
            
        self.train_labels = train_labels
        self.val_size = val_size
        
        
    def compute():
        pass
        


        
