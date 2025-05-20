try:
    from torch import optim, nn
    import lightning as L
    import torch.nn.functional as F
    from ._m3care import M3CareModel
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'Deep' needs to be installed."

LightningModuleBase = L.LightningModule if deepmodule_installed else object


class M3Care(LightningModuleBase):
    r"""

    Missing Modalities in Multimodal healthcare data (M3Care). [#m3carepaper]_ [#m3carecode]_

    M3Care is a multimodal classification framework that handles missing modalities by imputing latent
    task-relevant information using similar samples, based on a modality-adaptive similarity metric.
    It supports heterogeneous input types (e.g., tabular, text, vision).

    This class provides training, validation, testing, and prediction logic compatible with the Lightning Trainer.

    Parameters
    ----------
    input_dim : list of int, default=None
        A list specifying the input dimensions for each modality.
    hidden_dim : int, default=128
        Hidden dimension size.
    embed_size : int, default=128
        Size of the shared embedding space where modalities are projected.
    modalities : list of str, default=None
        Names of the modalities. Options are "tabular", "text" and "image".
    vocab : object, default=None
        Vocabulary object used for text modality preprocessing (if applicable).
    learning_rate : float, default=1e-4
        Learning rate for the optimizer.
    weight_decay : float, default=1e-4
        Weight decay used by the optimizer.
    output_dim : int, default=1
        Number of output dimensions. Typically 1 for binary classification.
    loss_fn : callable, default=None
        Loss function. If None, defaults to `nn.BCEWithLogitsLoss()`.
    keep_prob : float, default=0.5
        Dropout keep probability used in MLP layers.
    extractors : list of nn.Module, default=None
        List of custom feature extractors for each modality. If None, defaults will be used.

    References
    ----------
    .. [#m3carepaper] Zhang, Chaohe, et al. "M3care: Learning with missing modalities in multimodal healthcare data."
                      Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. 2022.
    .. [#m3carecode] https://github.com/choczhang/M3Care/

    Example
    --------
    >>> from imml.classify import M3Care
    >>> from lightning import Trainer
    >>> import torch
    >>> import numpy as np
    >>> import pandas as pd
    >>> from torch.utils.data import DataLoader
    >>> from imml.impute import get_observed_mod_indicator
    >>> from imml.load import M3CareDataset
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> train_data = M3CareDataset(Xs=[torch.from_numpy(X.values).float() for X in Xs],
                           observed_mod_indicator=torch.from_numpy(get_observed_mod_indicator(Xs).values),
                           y=torch.from_numpy(np.random.default_rng(42).integers(0, 2, len(Xs[0]))).float())
    >>> train_dataloader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
    >>> trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    >>> estimator = M3Care(modalities= ["tabular", "tabular"], input_dim=[X.shape[1] for X in Xs])
    >>> trainer.fit(estimator, train_dataloader)
    >>> trainer.predict(estimator, train_dataloader)
    """

    def __init__(self, input_dim: list = None, hidden_dim: int = 128, embed_size: int = 128, modalities: list = None,
                 vocab = None, learning_rate: float = 1e-4, weight_decay: float = 1e-4, output_dim: int = 1,
                 loss_fn: callable = None, keep_prob: float = 0.5, extractors: list = None):

        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        super().__init__()

        self.model = M3CareModel(input_dim=input_dim, hidden_dim=hidden_dim, embed_size=embed_size, vocab=vocab,
                                 modalities=modalities, output_dim=output_dim, keep_prob=keep_prob, extractors=extractors)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = loss_fn


    def training_step(self, batch, batch_idx):
        r"""
        Method required for training using Pytorch Lightning trainer.
        """
        Xs, y, observed_mod_indicator = batch
        y_pred, _ = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator)
        loss = self.loss_fn(y_pred.squeeze(), y)
        return loss


    def validation_step(self, batch, batch_idx):
        r"""
        Method required for validating using Pytorch Lightning trainer.
        """
        Xs, y, observed_mod_indicator = batch
        y_pred, _ = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator)
        loss = self.loss_fn(y_pred.squeeze(), y)
        return loss


    def test_step(self, batch, batch_idx):
        r"""
        Method required for testing using Pytorch Lightning trainer.
        """
        Xs, y, observed_mod_indicator = batch
        y_pred, _ = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator)
        loss = self.loss_fn(y_pred.squeeze(), y)
        return loss


    def predict_step(self, batch, batch_idx):
        r"""
        Method required for predicting using Pytorch Lightning trainer.
        """
        Xs, y, observed_mod_indicator = batch
        y_pred, _ = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator)
        loss = F.sigmoid(y_pred)
        return loss


    def configure_optimizers(self):
        r"""
        Method required for training using Pytorch Lightning trainer.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)