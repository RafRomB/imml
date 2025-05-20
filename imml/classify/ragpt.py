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

    Retrieval-AuGmented dynamic Prompt Tuning (RAGPT). [#ragptpaper]_ [#ragptcode]_

    RAGPT is designed for incomplete vision-language learning, where one modality may be missing at
    inference or training time. It combines three core modules to address this challenge: 1) Multi-Channel Retriever,
    which retrieves semantically similar instances from a training database, per modality; 2) Missing Modality
    Generator, which fills in missing modality data using context from retrieved neighbors; and 3) Context-Aware
    Prompter, which generates dynamic prompts based on context to improve downstream classification in
    multimodal transformers.

    This class provides training, validation, testing, and prediction logic compatible with the Lightning Trainer.

    Parameters
    ----------
    vilt : transformers.ViltModel, optional
        Pretrained model used for joint vision-language encoding. If None, defaults to
        ViltModel.from_pretrained('dandelin/vilt-b32-mlm').
    max_text_len : int, default=128
        Maximum number of tokens for text inputs.
    max_image_len : int, default=145
        Maximum number of image patches/tokens processed by the vision encoder.
    prompt_position : int, default=0
        Index position at which to insert dynamic prompts in the transformer input sequence.
    prompt_length : int, default=1
        Number of prompt tokens to insert for dynamic prompt tuning.
    dropout_rate : float, default=0.2
        Dropout probability.
    hidden_dim : int, default=768
        Hidden dimension size.
    cls_num : int, default=2
        Number of target classes for classification tasks.
    loss : callable, optional
        Loss function. If None, defaults to `F.cross_entropy`.
    learning_rate : float, default=1e-3
        Learning rate for the optimizer.
    weight_decay : float, default=2e-2
        Weight decay used by the optimizer.

    References
    ----------
    .. [#ragptpaper] Lang, J., Z. Cheng, T. Zhong, and F. Zhou. “Retrieval-Augmented Dynamic Prompt Tuning for
                     Incomplete Multimodal Learning”. Proceedings of the AAAI Conference on Artificial Intelligence,
                     vol. 39, no. 17, Apr. 2025, pp. 18035-43, doi:10.1609/aaai.v39i17.33984.
    .. [#ragptcode] https://github.com/Jian-Lang/RAGPT/

    Example
    --------
    >>> from imml.retrieve import MCR
    >>> from imml.load import RAGPTDataset
    >>> from imml.classify import RAGPT
    >>> from lightning import Trainer
    >>> from torch.utils.data import DataLoader
    >>> images = ["docs/figures/graph.png", "docs/figures/logo_imml.png",
                  "docs/figures/graph.png", "docs/figures/logo_imml.png"]
    >>> texts = ["This is the graphical abstract of iMML.", "This is the logo of iMML.",
                 "This is the graphical abstract of iMML.", "This is the logo of iMML."]
    >>> Xs = [images, texts]
    >>> y = [0, 1, 0, 1]
    >>> modalities = ["image", "text"]
    >>> estimator = MCR(modalities=modalities)
    >>> database = estimator.fit_transform(Xs=Xs, y=y)
    >>> train_data = RAGPTDataset(database=database)
    >>> train_dataloader = DataLoader(train_data)
    >>> trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    >>> estimator = RAGPT()
    >>> trainer.fit(estimator, train_dataloader)
    >>> trainer.predict(estimator, train_dataloader)
    """


    def __init__(self, vilt: ViltModel = None, max_text_len: int = 128, max_image_len: int = 145,
                 prompt_position: int = 0, prompt_length: int = 1, dropout_rate: float = 0.2, hidden_dim: int = 768,
                 cls_num: int = 2, loss: callable = None, learning_rate: float = 1e-3,
                 weight_decay: float = 2e-2):
        
        super().__init__()

        self.model = RAGPTModel(vilt=vilt, max_text_len=max_text_len, max_image_len=max_image_len,
                                prompt_position=prompt_position, prompt_length=prompt_length,
                                dropout_rate=dropout_rate, hidden_dim=hidden_dim, cls_num=cls_num)
        if loss is None:
            loss = F.cross_entropy
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