from . import FFNEncoder, RNNEncoder, TextEncoder, MML

try:
    import torch
    from torch import nn
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'Deep' needs to be installed."

nnModuleBase = nn.Module if deepmodule_installed else object


class MUSEModel(nnModuleBase):

    def __init__(self, input_dim: list = None, modalities: list = None, extractors: list = None, tokenizer=None,
                 hidden_dim: int = 128, cls_num: int = 2, gnn_layers: int = 2, gnn_norm: str = None,
                 code_pretrained_embedding: bool = True, bert_type: str = "prajjwal1/bert-tiny", dropout: float = 0.25
                 ):

        super().__init__()

        self.tokenizer = tokenizer
        self.modalities = modalities
        self.hidden_dim = hidden_dim
        self.code_pretrained_embedding = code_pretrained_embedding
        self.bert_type = bert_type
        self.dropout = dropout
        self.gnn_layers = gnn_layers
        self.gnn_norm = gnn_norm

        self.dropout_layer = nn.Dropout(dropout)

        if modalities is None:
            raise ValueError(f"Invalid modalities. It must be a list. A {type(modalities)} was passed.")
        if extractors is None:
            extractors = [None] * len(modalities)
        if input_dim is not None:
            self.input_dim = iter(input_dim)

        for i, (mod, extractor) in enumerate(zip(self.modalities, extractors)):
            if mod == "tabular":
                if extractor is None:
                    encoder = FFNEncoder(input_dim=next(self.input_dim), hidden_dim=hidden_dim,
                                         output_dim=hidden_dim, dropout_prob=dropout, num_layers=2)
                    extractor = nn.Sequential(encoder, nn.Linear(hidden_dim, hidden_dim))
            elif mod == "series":
                if extractor is None:
                    encoder = RNNEncoder(input_size=158, hidden_size=hidden_dim, num_layers=1, rnn_type="GRU",
                                         dropout=dropout, bidirectional=False)
                    extractor = nn.Sequential(encoder, nn.Linear(hidden_dim, hidden_dim))
            elif mod == "text":
                if extractors is None:
                    encoder = TextEncoder(bert_type)
                    for param in encoder.parameters():
                        param.requires_grad = False
                    output_dim = encoder.model.config.hidden_size
                    extractor = nn.Sequential(encoder, nn.Linear(output_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown modality type: {mod}")
            setattr(self, f"extractor{i}", extractor)

        self.mml = MML(num_modalities=len(modalities), hidden_channels=hidden_dim, num_layers=gnn_layers,
                       dropout=dropout, normalize_embs=gnn_norm, num_classes=cls_num)


    def forward(self, Xs, y, observed_mod_indicator, y_indicator):
        observed_mod_indicator = ~observed_mod_indicator
        transformed_Xs = []
        for X_idx, (X,mod) in enumerate(zip(Xs, self.modalities)):
            extractor = getattr(self, f"extractor{X_idx}")
            code_embedding = extractor(X)
            code_embedding[observed_mod_indicator[:,X_idx]] = 0
            code_embedding = self.dropout_layer(code_embedding)
            transformed_Xs.append(code_embedding)
        loss = self.mml(Xs=transformed_Xs, observed_mod_indicator=observed_mod_indicator, y=y, y_indicator=y_indicator)
        return loss


    def predict(self, Xs, observed_mod_indicator):
        observed_mod_indicator = ~observed_mod_indicator
        transformed_Xs = []
        for X_idx, (X,mod) in enumerate(zip(Xs, self.modalities)):
            extractor = getattr(self, f"extractor{X_idx}")
            code_embedding = extractor(X)
            code_embedding[observed_mod_indicator[:,X_idx]] = 0
            code_embedding = self.dropout_layer(code_embedding)
            transformed_Xs.append(code_embedding)
        logits = self.mml.inference(Xs=transformed_Xs, observed_mod_indicator=observed_mod_indicator)
        return logits