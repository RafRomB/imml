
try:
    from torch import optim, nn
    import lightning as L
    import torch.nn.functional as F
    from ._muse import MUSEModel
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'Deep' needs to be installed."

LightningModuleBase = L.LightningModule if deepmodule_installed else object


class MUSE(LightningModuleBase):
    r"""

    Mutual-consistent graph contrastive learning (MUSE). [#musepaper]_ [#musecode]_

    MUSE is a multimodal representation learning framework designed to handle missing modalities and partially
    labeled data. It uses a bipartite graph between samples and modalities to support arbitrary missingness patterns
    and a mutual-consistent contrastive loss to encourage the learning of label-discriminative, modality-consistent
    features.

    This class provides training, validation, testing, and prediction logic compatible with the Lightning Trainer.

    Parameters
    ----------
    input_dim : list of int, default=None
        A list specifying the input dimensions for each modality.
    hidden_dim : int, default=128
        Hidden dimension size.
    modalities : list of str, default=None
        Names of the modalities. Options are "tabular", "text" and "image".
    tokenizer : str, default=None
        Tokenizer to use for text modality. If None, defaults to "emilyalsentzer/Bio_ClinicalBERT" tokenizer.
    learning_rate : float, default=2e-4
        Learning rate for the optimizer.
    weight_decay : float, default=0
        Weight decay used by the optimizer.
    cls_num : int, default=2
        Number of output classes for the classification task.
    extractors : list of nn.Module, default=None
        List of custom feature extractors for each modality. If None, defaults will be used.
    gnn_layers : int, default=2
        Number of GNN layers used to propagate sample-modality representations.
    gnn_norm : str or None, default=None
        Optional normalization strategy in GNN layers (e.g., 'batchnorm', 'layernorm').
    code_pretrained_embedding : bool, default=True
        If True, initializes pretrained embeddings for text/code features.
    bert_type : str, default="prajjwal1/bert-tiny"
        HuggingFace model name or path for BERT backbone used in the text encoder.
    dropout : float, default=0.25
        Dropout rate applied in the encoders and classifier head.

    References
    ----------
    .. [#musepaper] Wu, Zhenbang, et al. "Multimodal patient representation learning with missing modalities and
                    labels." The Twelfth International Conference on Learning Representations. 2024.
    .. [#musecode] https://github.com/zzachw/MUSE/

    Example
    --------
    >>> from imml.classify import MUSE
    >>> from lightning import Trainer
    >>> import torch
    >>> import numpy as np
    >>> import pandas as pd
    >>> from torch.utils.data import DataLoader
    >>> from imml.load import MUSEDataset
    >>> from imml.impute import get_observed_mod_indicator
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> train_data = MUSEDataset(Xs=[torch.from_numpy(X.values).float() for X in Xs],
                                 observed_mod_indicator=torch.from_numpy(get_observed_mod_indicator(Xs).values),
                                 y=torch.from_numpy(np.random.default_rng(42).integers(0, 2, len(Xs[0]))).float(),
                                 y_indicator=torch.ones((len(Xs[0]))).bool()
                                 )
    >>> train_dataloader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
    >>> trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    >>> estimator = MUSE(modalities= ["tabular", "tabular"], input_dim=[X.shape[1] for X in Xs])
    >>> trainer.fit(estimator, train_dataloader)
    >>> trainer.predict(estimator, train_dataloader)
    """

    def __init__(self, input_dim: list = None, hidden_dim: int = 128, modalities: list = None,
                 tokenizer=None, learning_rate: float = 2e-4, weight_decay: float = 0, cls_num: int = 2,
                 extractors: list = None, gnn_layers: int = 2, gnn_norm: str = None,
                 code_pretrained_embedding: bool = True, bert_type: str = "prajjwal1/bert-tiny", dropout: float = 0.25):

        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        super().__init__()

        self.model = MUSEModel(input_dim=input_dim, tokenizer=tokenizer, hidden_dim=hidden_dim,
                               modalities=modalities, cls_num=cls_num, extractors=extractors,
                               gnn_layers=gnn_layers, gnn_norm=gnn_norm, bert_type=bert_type, dropout=dropout,
                               code_pretrained_embedding=code_pretrained_embedding)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


    def training_step(self, batch, batch_idx):
        r"""
        Method required for training using Pytorch Lightning trainer.
        """
        Xs, y, observed_mod_indicator, y_indicator = batch
        loss = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator, y=y, y_indicator=y_indicator)
        return loss


    def validation_step(self, batch, batch_idx):
        r"""
        Method required for validating using Pytorch Lightning trainer.
        """
        Xs, y, observed_mod_indicator, y_indicator = batch
        loss = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator, y=y, y_indicator=y_indicator)
        return loss


    def test_step(self, batch, batch_idx):
        r"""
        Method required for testing using Pytorch Lightning trainer.
        """
        Xs, y, observed_mod_indicator, y_indicator = batch
        loss = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator, y=y, y_indicator=y_indicator)
        return loss


    def predict_step(self, batch, batch_idx):
        r"""
        Method required for predicting using Pytorch Lightning trainer.
        """
        Xs, y, observed_mod_indicator, y_indicator = batch
        pred = self.model.predict(Xs=Xs, observed_mod_indicator=observed_mod_indicator)
        return pred


    def configure_optimizers(self):
        r"""
        Method required for training using Pytorch Lightning trainer.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)