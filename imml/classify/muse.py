from ._muse import MUSEModel

try:
    from torch import optim, nn
    import lightning as L
    import torch.nn.functional as F
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'Deep' needs to be installed."

LightningModuleBase = L.LightningModule if deepmodule_installed else object


class MUSE(LightningModuleBase):
    r"""

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
    .. [#mappaper] Lee, Yi-Lun, et al. "Multimodal prompting with missing modalities for visual recognition."
                   Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    .. [#mapcode] https://github.com/YiLunLee/missing_aware_prompts
    .. [#viltpaper] Kim, Wonjae, Bokyung Son, and Ildoo Kim. "Vilt: Vision-and-language transformer without
                    convolution or region supervision." International conference on machine learning. PMLR, 2021.
    .. [#viltcode] https://github.com/dandelin/ViLT

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
        Xs, y, observed_mod_indicator, y_indicator = batch
        loss = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator, y=y, y_indicator=y_indicator)
        return loss


    def validation_step(self, batch, batch_idx):
        Xs, y, observed_mod_indicator, y_indicator = batch
        loss = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator, y=y, y_indicator=y_indicator)
        return loss


    def test_step(self, batch, batch_idx):
        Xs, y, observed_mod_indicator, y_indicator = batch
        loss = self.model(Xs=Xs, observed_mod_indicator=observed_mod_indicator, y=y, y_indicator=y_indicator)
        return loss


    def predict_step(self, batch, batch_idx):
        Xs, y, observed_mod_indicator, y_indicator = batch
        pred = self.model.predict(Xs=Xs, observed_mod_indicator=observed_mod_indicator)
        return pred


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)