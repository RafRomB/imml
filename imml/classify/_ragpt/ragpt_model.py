from .vilt import ViltModel
from .modules import MMG, CAP

try:
    import torch
    from torch import nn
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'Deep' needs to be installed."

nnModuleBase = nn.Module if deepmodule_installed else object
ViltModel = ViltModel if deepmodule_installed else object


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class RAGPTModel(nnModuleBase):
    def __init__(self, vilt: ViltModel = None, max_text_len: int = 128, max_image_len: int = 145,
                 prompt_position: int = 0, prompt_length: int = 1, dropout_rate: float = 0.2,
                 hidden_dim: int = 768, cls_num: int = 2):

        super().__init__()

        if vilt is None:
            vilt = ViltModel.from_pretrained('dandelin/vilt-b32-mlm')

        self.max_text_len = max_text_len
        self.embedding_layer = vilt.embeddings
        self.encoder_layer = vilt.encoder.layer
        self.layernorm = vilt.layernorm
        self.prompt_length = prompt_length
        self.prompt_position = prompt_position
        self.hs = hidden_dim

        self.freeze()
        self.pooler = vilt.pooler
        self.MMG_t = MMG(n = max_text_len, d=hidden_dim, dropout_rate=dropout_rate)
        self.MMG_i = MMG(n = max_image_len, d=hidden_dim, dropout_rate=dropout_rate)
        self.dynamic_prompt = CAP(prompt_length=prompt_length)
        self.label_enhanced = nn.Parameter(torch.randn(cls_num, hidden_dim))
        self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        self.classifier.apply(init_weights)

    def freeze(self):
        for param in self.embedding_layer.parameters():
            param.requires_grad = False
        for param in self.encoder_layer.parameters():
            param.requires_grad = False
        for param in self.layernorm.parameters():
            param.requires_grad = False

    def forward(self,
                input_ids: torch.Tensor,
                pixel_values: torch.Tensor,
                pixel_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                r_t_list: torch.Tensor, 
                r_i_list: torch.Tensor,
                r_l_list: torch.Tensor,
                observed_image = None,
                observed_text = None,
                image_token_type_idx=1):
        embedding, attention_mask = self.embedding_layer(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids,
                                                         inputs_embeds=None,
                                                         image_embeds=None,
                                                         pixel_values=pixel_values,
                                                         pixel_mask=pixel_mask,
                                                         image_token_type_idx=image_token_type_idx)
        text_emb = embedding[:, :self.max_text_len, :]
        image_emb = embedding[:, self.max_text_len:, :]

        recovered_t = self.MMG_t(r_t_list)
        recovered_i = self.MMG_i(r_i_list)
        t_observed_mask = torch.tensor(observed_text).to(pixel_values.device)
        i_observed_mask = torch.tensor(observed_image).to(pixel_values.device)
        observed_mask_t = t_observed_mask.view(-1, 1, 1).expand(-1,self.max_text_len, self.hs)
        observed_mask_i = i_observed_mask.view(-1, 1, 1).expand(-1, 145, self.hs)
        text_emb = text_emb * observed_mask_t + recovered_t * (~observed_mask_t)
        image_emb = image_emb * observed_mask_i + recovered_i * (~observed_mask_i)
        
        t_prompt,i_prompt = self.dynamic_prompt(r_i=r_i_list, r_t=r_t_list, T=text_emb, V=image_emb)
        t_prompt = torch.mean(t_prompt, dim=1)
        i_prompt = torch.mean(i_prompt, dim=1)

        label_emb = self.label_enhanced[r_l_list]
        label_cls = self.label_enhanced
        label_emb = torch.mean(label_emb, dim=1)
        label_emb = label_emb.view(-1, 1, self.hs)

        output = torch.cat([text_emb, image_emb], dim=1)
        for i, layer_module in enumerate(self.encoder_layer):
            if i == self.prompt_position:
                output = torch.cat([label_emb,t_prompt.unsqueeze(1),i_prompt.unsqueeze(1),output], dim=1)
                N = embedding.shape[0]
                attention_mask = torch.cat([torch.ones(N,1+self.prompt_length*2).to(pixel_values.device), attention_mask],
                                           dim=1)
                layer_outputs = layer_module(output, attention_mask=attention_mask)
                output = layer_outputs[0]
            else:
                layer_outputs = layer_module(output, attention_mask=attention_mask)
                output = layer_outputs[0]
        output = self.layernorm(output)
        output = self.pooler(output)
        output = torch.cat([output,label_emb.squeeze(1)],dim=1)
        output = self.classifier(output)
        label_cls = label_cls.repeat(N, 1,1)
        label_cls = label_cls.transpose(-1,-2)
        output = output.unsqueeze(1)
        output = torch.matmul(output, label_cls)
        output = output.squeeze(1)
        return output