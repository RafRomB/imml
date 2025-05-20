from ._ragpt import RAGPTModel
from ._ragpt.vilt import ViltModel

try:
    import torch
    import torch.nn.functional as F
    import lightning as L
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'Deep' needs to be installed."

LightningModuleBase = L.LightningModule if deepmodule_installed else object


class RAGPT(LightningModuleBase):
    r"""

    Retrieval-AuGmented dynamic Prompt Tuning (RAGPT) [#ragptpaper]_ [#ragptcode]_

    RAGPT comprises three modules: (I) the multi-channel retriever, which identifies similar instances through a
    within-modality retrieval strategy, (II) the missing modality generator, which recovers missing information
    using retrieved contexts, and (III) the context-aware prompter, which captures contextual knowledge from
    relevant instances and generates dynamic prompts to largely enhance the MMT’s robustness.

    Parameters
    ----------
    bert_config : BertConfig
        Configuration object for the BERT model used to process the text modality.
    classifier : nn.Sequential
        A neural network classifier that maps the fused multimodal representation to output predictions.
    model : str, default="vit_base_patch32_384"
        Name of the vision model backbone (e.g., a Vision Transformer variant).
    load_path : str, default=""
        Path to a pretrained checkpoint to load the model weights from. You can download the pre-trained ViLT model
        weights from https://github.com/dandelin/ViLT.
    test_only : bool, default=False
        Whether to run the model in evaluation-only mode without training.
    finetune_first : bool, default=False
        Whether to finetune only the backbone initially before training prompts.
    prompt_type : str, default="input"
        Type of prompt injection. One of ['input', 'attention'].
    prompt_length : int, default=16
        Number of prompt tokens.
    learnt_p : bool, default=True
        If True, the prompt embeddings are learnable parameters.
    prompt_layers : list, default=None
        List of layer indices. If None, prompt_layers is [0,1,2,3,4,5].
    multi_layer_prompt : bool, default=True
        If True, prompts are injected into multiple layers of the transformer model.
    loss_name : str, default="accuracy"
        Name of loss functions to be used during training. One of ["accuracy", "F1_scores", "AUROC"]
    learning_rate : float, default=1e-2
        Initial learning rate for the optimizer.
    weight_decay : float, default=2e-2
        Weight decay for the optimizer.
    lr_mult : float, default=1
        Multiplier applied to lr for downstream heads.
    end_lr : float, default=0
        Final learning rate after polynomial decay.
    decay_power : float, default=1
        Power for polynomial learning rate decay scheduling.
    optim_type : str, default="adamw"
        Optimizer type to use (e.g., 'adamw', 'sgd').
    warmup_steps : int, default=2500
        Number of warm-up steps for the learning rate scheduler.

    References
    ----------
    .. [#ragptpaper] Lang, J., Z. Cheng, T. Zhong, and F. Zhou. “Retrieval-Augmented Dynamic Prompt Tuning for
                     Incomplete Multimodal Learning”. Proceedings of the AAAI Conference on Artificial Intelligence,
                     vol. 39, no. 17, Apr. 2025, pp. 18035-43, doi:10.1609/aaai.v39i17.33984.
    .. [#ragptcode] https://github.com/Jian-Lang/RAGPT/

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


    def __init__(self, vilt: ViltModel = None, max_text_len: int = 128, max_image_len: int = 145,
                 prompt_position: int = 0, prompt_length: int = 1, dropout_rate: float = 0.2, hidden_dim: int = 768,
                 cls_num: int = 2, loss: callable = F.cross_entropy, learning_rate: float = 1e-3,
                 weight_decay: float = 2e-2):
        
        super().__init__()

        self.model = RAGPTModel(vilt=vilt, max_text_len=max_text_len, max_image_len=max_image_len,
                                prompt_position=prompt_position, prompt_length=prompt_length,
                                dropout_rate=dropout_rate, hidden_dim=hidden_dim, cls_num=cls_num)
        self.loss = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


    def training_step(self, batch, batch_idx):
        r"""
        Method required for training using Pytorch Lightning trainer.
        """
        labels = batch.pop('label').long()
        preds = self.model(**batch)
        loss = self.loss(preds, labels)
        return loss


    def validation_step(self, batch, batch_idx):
        r"""
        Method required for validating using Pytorch Lightning trainer.
        """
        labels = batch.pop('label').long()
        preds = self.model(**batch)
        loss = self.loss(preds, labels)
        return loss


    def test_step(self, batch, batch_idx):
        r"""
        Method required for testing using Pytorch Lightning trainer.
        """
        labels = batch.pop('label').long()
        preds = self.model(**batch)
        loss = self.loss(preds, labels)
        return loss


    def predict_step(self, batch, batch_idx):
        r"""
        Method required for predicting using Pytorch Lightning trainer.
        """
        _ = batch.pop('label').long()
        preds = self.model(**batch)
        return preds


    def configure_optimizers(self):
        r"""
        Method required for training using Pytorch Lightning trainer.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer