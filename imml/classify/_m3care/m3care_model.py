import numpy as np

from . import (GraphConvolution, MM_transformer_encoder, PositionalEncoding, guassian_kernel,
               init_weights, length_to_mask, clones, NMT_tran)

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    import torchvision.models as models
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'Deep' needs to be installed."

nnModuleBase = nn.Module if deepmodule_installed else object


class M3CareModel(nnModuleBase):

    def __init__(self, input_dim: list = None, hidden_dim: int = 128, embed_size: int = 128, modalities: list = None,
                 vocab = None, output_dim=1, keep_prob=1, extractors: list = None):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.modalities = modalities
        self.output_dim = output_dim
        self.keep_prob = keep_prob
        self.n_mods = len(modalities)

        if modalities is None:
            raise ValueError(f"Invalid modalities. It must be a list. A {type(modalities)} was passed.")
        if extractors is None:
            extractors = [None] * len(modalities)
        if input_dim is not None:
            self.input_dim = iter(input_dim)

        for i, (mod, extractor) in enumerate(zip(self.modalities, extractors)):
            if mod == "tabular":
                if extractor is None:
                    extractor = nn.Linear(next(self.input_dim), hidden_dim)
            elif mod == "text":
                if extractor is None:
                    extractor = NMT_tran(embed_size=embed_size, hidden_size=hidden_dim,
                                         dropout_rate=1 - self.keep_prob, vocab=vocab)
            elif mod == "image":
                if extractors is None:
                    extractor = nn.Sequential(models.resnet18(),
                                              nn.Linear(1000, self.hidden_dim)
                                              )
            else:
                raise ValueError(f"Unknown modality type: {mod}")
            setattr(self, f"extractor{i}", extractor)

        self.MM_model1 = MM_transformer_encoder(input_dim=self.hidden_dim, d_model=self.hidden_dim, \
                                               MHD_num_head=4, d_ff=self.hidden_dim * 4, output_dim=1)
        self.MM_model2 = MM_transformer_encoder(input_dim=self.hidden_dim, d_model=self.hidden_dim, \
                                                MHD_num_head=1, d_ff=self.hidden_dim * 4, output_dim=1)

        self.token_type_embeddings = nn.Embedding(6, self.hidden_dim)
        self.token_type_embeddings.apply(init_weights)
        self.PositionalEncoding = PositionalEncoding(self.hidden_dim, dropout=0, max_len=5000)

        self.dropout = nn.Dropout(p=1 - self.keep_prob)

        self.proj1 = nn.Linear(self.hidden_dim * (len(self.modalities)+1), self.hidden_dim * 2)
        self.out_layer = nn.Linear(self.hidden_dim * 2, self.output_dim)

        self.threshold = nn.Parameter(torch.ones(size=(1,)) + 1)
        self.simiProj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.selu = nn.SELU()

        self.bn = nn.BatchNorm1d(self.hidden_dim)

        self.simiProj = clones(torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True), nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True), nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
        ), self.n_mods)

        self.GCN1 = clones(GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True), self.n_mods)
        self.GCN2 = clones(GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True), self.n_mods)
        self.weight1 = clones(nn.Linear(self.hidden_dim, 1), self.n_mods)
        self.weight2 = clones(nn.Linear(self.hidden_dim, 1), self.n_mods)
        self.eps = nn.ParameterList([nn.Parameter(torch.ones(1)+1) for _ in range(self.n_mods)])


    def forward(self, Xs, observed_mod_indicator):

        hidden00 = []
        mask_mats = []
        mask2_mats = []
        for X_idx, (X,mod) in enumerate(zip(Xs, self.modalities)):
            extractor = getattr(self, f"extractor{X_idx}")
            if mod == 'tabular':
                feat = extractor(X)
                mask_mat = observed_mod_indicator[:, X_idx]
            elif mod == 'image':
                feat = extractor(X)
                mask_mat = observed_mod_indicator[:, X_idx]
            elif mod == 'text':
                feat, lens = extractor(X)
                feat = feat[:, 0]
                mask_mat = torch.from_numpy(np.array(lens)).to(feat.device)
            else:
                raise ValueError(f"Unknown modality type: {mod}")
            feat = F.relu(feat)
            mask_mat = length_to_mask(mask_mat.int()).int()
            mask2 = mask_mat * mask_mat.permute(1,0)
            hidden00.append(feat)
            mask_mats.append(mask_mat)
            mask2_mats.append(mask2)

        sim_mats = []
        diffs = []
        for i, h in enumerate(hidden00):
            h0 = h
            p = F.relu(self.simiProj[i](h0))
            km1 = guassian_kernel(self.bn(p), kernel_mul=2.0, kernel_num=3)
            km2 = guassian_kernel(self.bn(h0), kernel_mul=2.0, kernel_num=3)
            sim = ((1 - torch.sigmoid(self.eps[i])) * km1 + torch.sigmoid(self.eps[i]) * km2)
            sim = sim * mask_mats[i]
            sim_mats.append(sim)
            diff = torch.abs(torch.norm(self.simiProj[i](h), dim=1) - torch.norm(h, dim=1))
            diffs.append(diff)

        sum_of_diff = torch.stack(diffs, dim=1).sum(dim=1)

        sim_sum = torch.stack(sim_mats, dim=0).sum(dim=0)
        mask_sum = torch.stack(mask2_mats, dim=0).sum(dim=0)
        sim_avg = sim_sum / mask_sum

        th = torch.sigmoid(self.threshold)[0]
        sim_th = F.relu(sim_avg - th)
        bin_mask = sim_th > 0
        sim_final = sim_th + bin_mask * th.detach()

        final_h = []
        gs = []
        for i, (h,mask2) in enumerate(zip(hidden00, mask2_mats)):
            g = F.relu(self.GCN1[i](sim_final*mask2, h))
            g = F.relu(self.GCN2[i](sim_final*mask2, g))
            gs.append(g)
            w1 = torch.sigmoid(self.weight1[i](g))
            w2 = torch.sigmoid(self.weight2[i](h))
            w1 = w1 / (w1 + w2)
            w2 = 1 - w1
            final = w1 * g + w2 * h
            final_h.append(final)

        embs = []
        batch_size = hidden00[0].size(0)
        for idx, (h, mask) in enumerate(zip(hidden00, mask_mats)):
            emb = self.PositionalEncoding(h.unsqueeze(1))
            emb = emb + self.token_type_embeddings(torch.full((batch_size,1), idx, dtype=torch.long, device=h.device))
            embs.append(emb)

        z0 = torch.cat(embs, dim=1)
        z0_mask = torch.cat(mask_mats, dim=-1).int()
        z1 = F.relu(self.MM_model1(z0, z0_mask.unsqueeze(1)))
        z2 = F.relu(self.MM_model2(z1, z0_mask.unsqueeze(1)))
        combined_hidden = torch.cat([z2[:,0,:]]+final_h, dim=-1)
        last_hs_proj = self.dropout(F.relu(self.proj1(combined_hidden)))
        output = self.out_layer(last_hs_proj)

        return output, sum_of_diff

