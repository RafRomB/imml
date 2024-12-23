import pytest
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from imml.ampute import Amputer
from imml.preprocessing import MultiModTransformer
from imml.cluster import MRGCN
from imml.load import MRGCNDataset

try:
    import torch
    from torch import nn
    import lightning.pytorch as pl
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from lightning import Trainer
    torch_installed = True
except ImportError:
    torch_installed = False


@pytest.fixture
def sample_data():
    if torch_installed:
        X = np.random.default_rng(42).random((25, 10))
        X1, X2, X3 = X[:, :3], X[:, 3:5], X[:, 5:]
        Xs_numpy = [X1, X2, X3]
        Xs_torch = [torch.from_numpy(X.astype(np.float32)) for X in Xs_numpy]
        return Xs_torch, Xs_numpy

def test_pytorch_not_installed(sample_data):
    if torch_installed:
        MRGCN(Xs=sample_data[0])
    else:
        with pytest.raises(ImportError, match="torch and lightning needs to be installed."):
            MRGCN(Xs=sample_data[0])

def test_default_params(sample_data):
    n_clusters = 3
    if torch_installed:
        Xs = sample_data[0]
        n_samples = len(Xs[0])
        train_data = MRGCNDataset(Xs=Xs)
        train_dataloader = DataLoader(dataset=train_data, batch_size=n_samples, shuffle=True)
        trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
        estimator = MRGCN(Xs=Xs, n_clusters=n_clusters)
        trainer.fit(estimator, train_dataloader)
        train_dataloader = DataLoader(dataset=train_data, batch_size=n_samples, shuffle=False)
        labels = trainer.predict(estimator, train_dataloader)[0]
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == estimator.n_clusters
        assert min(labels) == 0
        assert max(labels) == (estimator.n_clusters - 1)
        assert not np.isnan(labels).any()
        embedding_ = estimator._embedding(batch=Xs).detach().cpu().numpy()
        assert len(embedding_) == n_samples

def test_invalid_params(sample_data):
    estimator = MRGCN
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters='invalid', Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters=0, Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid Xs."):
        estimator(Xs="invalid")
    with pytest.raises(ValueError, match="Invalid k_num."):
        estimator(k_num="invalid", Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid k_num."):
        estimator(k_num=0, Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(learning_rate="invalid", Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(learning_rate=0, Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(learning_rate=0, Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid reg2."):
        estimator(reg2="invalid", Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid reg3."):
        estimator(reg3="invalid", Xs=sample_data[0])

def test_missing_values_handling(sample_data):
    n_clusters = 2
    if torch_installed:
        Xs = sample_data[1]
        n_samples = len(Xs[0])
        amputer = Amputer(p= 0.3, random_state=42)
        imputer = SimpleImputer(strategy="constant", fill_value=0.0).set_output(transform="pandas")
        transformer = FunctionTransformer(lambda x: torch.from_numpy(x.values.astype(np.float32)))
        pipeline = make_pipeline(amputer, MultiModTransformer(imputer), MultiModTransformer(transformer))
        transformed_Xs = pipeline.fit_transform(Xs)
        train_data = MRGCNDataset(Xs=transformed_Xs)
        train_dataloader = DataLoader(dataset=train_data, batch_size=len(transformed_Xs[0]), shuffle=True)
        trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
        estimator = MRGCN(Xs=transformed_Xs, n_clusters=n_clusters)
        trainer.fit(estimator, train_dataloader)
        train_dataloader = DataLoader(dataset=train_data, batch_size=len(transformed_Xs[0]), shuffle=False)
        labels = trainer.predict(estimator, train_dataloader)[0]
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == estimator.n_clusters
        assert min(labels) == 0
        assert max(labels) == (estimator.n_clusters - 1)
        assert not np.isnan(labels).any()
        embedding_ = estimator._embedding(batch=transformed_Xs).detach().cpu().numpy()
        assert not np.isnan(embedding_).any().any()
        assert len(embedding_) == n_samples


if __name__ == "__main__":
    pytest.main()