import os
import numpy as np
from PIL import Image
import pandas as pd

from .vilt import ViltModel, ViltImageProcessor

try:
    import torch
    from torch import nn
    import lightning as L
    from transformers import BertTokenizer
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'Deep' needs to be installed."

nnModuleBase = nn.Module if deepmodule_installed else object


class MemoryBankGenerator(nnModuleBase):
    def __init__(self, cfg):
        super(MemoryBankGenerator, self).__init__()
        pretrained_vilt = ViltModel.from_pretrained('dandelin/vilt-b32-mlm')
        self.embedding_layer = pretrained_vilt.embeddings
        self._freeze()
        self.tokenizer = BertTokenizer.from_pretrained('dandelin/vilt-b32-mlm', do_lower_case=True)
        self.image_processor = ViltImageProcessor.from_pretrained('dandelin/vilt-b32-mlm')
        self.dataset = cfg.data_para.dataset
        self.max_text_len = cfg.data_para.max_text_len
        self.max_image_len = cfg.data_para.max_image_len
        self.df_train = pd.read_pickle(rf'dataset/{self.dataset}/train.pkl')
        self.df_test = pd.read_pickle(rf'dataset/{self.dataset}/test.pkl')
        if self.dataset != "food101":
            self.df_valid = pd.read_pickle(rf'dataset/{self.dataset}/valid.pkl')
        self.batch_size = cfg.batch_size
        # check if memory_bank folder exists
        if not os.path.exists(f'dataset/memory_bank/{self.dataset}/text'):
            os.makedirs(f'dataset/memory_bank/{self.dataset}/text')
        if not os.path.exists(f'dataset/memory_bank/{self.dataset}/image'):
            os.makedirs(f'dataset/memory_bank/{self.dataset}/image')

    def _freeze(self):
        for param in self.embedding_layer.parameters():
            param.requires_grad = False

    def _encode(self, input_ids, pixel_values, pixel_mask, token_type_ids, attention_mask, image_token_type_idx=1):
        embedding, attention_mask = self.embedding_layer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            image_embeds=None,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            image_token_type_idx=image_token_type_idx
        )
        return embedding

    def _resize_image(self, img, size=(384, 384)):
        return img.resize(size, Image.BILINEAR)

    def _process_batch(self, df, start_idx, end_idx):
        texts = df['text'][start_idx:end_idx]
        ids = df['item_id'][start_idx:end_idx]

        text_encodings = self.tokenizer(
            texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        input_ids = text_encodings['input_ids']
        attention_mask = text_encodings['attention_mask']
        token_type_ids = text_encodings['token_type_ids']

        images = []
        for id in ids:
            if self.dataset == "hatememes":
                image_path = fr'dataset/{self.dataset}/image/{id}.png'
            elif self.dataset == "food101":
                image_path = fr'dataset/{self.dataset}/image/{id}.jpg'
            elif self.dataset == "mmimdb":
                image_path = fr'dataset/{self.dataset}/image/{id}.jpeg'
            image = Image.open(image_path).convert("RGB")
            image = self._resize_image(image)
            images.append(image)
        
        encoding_image_processor = self.image_processor(images, return_tensors="pt")
        pixel_values = encoding_image_processor["pixel_values"]
        pixel_mask = encoding_image_processor["pixel_mask"]

        emb = self._encode(input_ids, pixel_values, pixel_mask, token_type_ids, attention_mask)
        text_emb = emb[:, :self.max_text_len]
        image_emb = emb[:, self.max_text_len:]

        for i, id in enumerate(ids):
            np.save(f'dataset/memory_bank/{self.dataset}/text/{id}.npy', text_emb[i].detach().numpy())
            np.save(f'dataset/memory_bank/{self.dataset}/image/{id}.npy', image_emb[i].detach().numpy())
    
    def run(self):
        for i in range(0, len(self.df_train), self.batch_size):
            start_idx = i
            end_idx = min(i + self.batch_size, len(self.df_train))
            self._process_batch(self.df_train, start_idx, end_idx)


def resize_image(img, size=(384, 384)):
    return img.resize(size, Image.BILINEAR)

def get_collator(max_text_len, **kargs):
    collator = Collator(max_text_len, **kargs)
    return collator

class Collator:
    def __init__(self, max_text_len, **kargs):
        self.image_processor = ViltImageProcessor.from_pretrained('dandelin/vilt-b32-mlm')
        self.tokenizer = BertTokenizer.from_pretrained('dandelin/vilt-b32-mlm', do_lower_case=True)
        self.max_text_len = max_text_len

    def __call__(self, batch):
        text = [item['text'] for item in batch]
        image = [item['image'] for item in batch]
        label = [item['label'] for item in batch]
        r_t_list = [item['r_t_list'] for item in batch]
        r_i_list = [item['r_i_list'] for item in batch]
        observed_mask = [item['observed_mask'] for item in batch]
        r_l_list = [item['r_l_list'] for item in batch]
        text_encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        input_ids = text_encoding['input_ids']
        attention_mask = text_encoding['attention_mask']
        token_type_ids = text_encoding['token_type_ids']
        image = [resize_image(img) for img in image]
        image_encoding = self.image_processor(image, return_tensors="pt")
        pixel_values = image_encoding["pixel_values"]
        pixel_mask = image_encoding["pixel_mask"]
        input_ids = torch.tensor(input_ids,dtype=torch.int64)
        token_type_ids = torch.tensor(token_type_ids,dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask,dtype=torch.int64)
        label = torch.tensor(label,dtype=torch.float)
        r_l_list = torch.tensor(r_l_list,dtype=torch.long)
        r_t_list = torch.tensor(r_t_list,dtype=torch.float)
        r_i_list = torch.tensor(r_i_list,dtype=torch.float)
        return {
            "input_ids": torch.tensor(input_ids,dtype=torch.int64),
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "label": label,
            "r_t_list": r_t_list,
            "r_i_list": r_i_list,
            "observed_mask": torch.tensor(observed_mask,dtype=torch.int64),
            "r_l_list": r_l_list
        }
