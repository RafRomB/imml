from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from ..classify._ragpt.core_tools import resize_image
from ..classify._ragpt.vilt import ViltImageProcessor

try:
    import lightning.pytorch as pl
    from transformers import BertTokenizer
    import torch
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'Deep' needs to be installed."

TorchDatasetBase = torch.utils.data.Dataset if deepmodule_installed else object


class RAGPTDataset(TorchDatasetBase):


    def __init__(self, database: pd.DataFrame, max_text_len: int = 128):

        super().__init__()

        self.max_text_len = max_text_len
        self.img_path_list = database['img_path'].tolist()
        self.text_list = database['text'].tolist()
        self.label_list = database['label'].tolist()
        self.i2i_list = database['i2i_id_list'].tolist()
        self.t2t_list = database['t2t_id_list'].tolist()
        self.prompt_image_path = database['prompt_image_path'].tolist()
        self.prompt_text_path = database['prompt_text_path'].tolist()
        self.i2i_r_l_list_list = database['i2i_label_list'].tolist()
        self.t2t_r_l_list_list = database['t2t_label_list'].tolist()
        self.observed_image = database['observed_image'].tolist()
        self.observed_text = database['observed_text'].tolist()


    def __getitem__(self, index):
        text = self.text_list[index]
        image = self.img_path_list[index]
        image = Image.open(image) if pd.notna(image) else Image.new("RGBA", (256, 256), (0, 0, 0))
        image = image.convert("RGB")
        label = self.label_list[index]
        observed_text = self.observed_text[index]
        observed_image = self.observed_image[index]
        prompt_image_path = self.prompt_image_path[index]
        prompt_text_path = self.prompt_text_path[index]
        r_i_list = []
        r_t_list = []

        if (observed_text == 0) and (observed_image == 1):
            text = "I love deep learning" * 1024
            r_l_list = self.i2i_r_l_list_list[index]
            for i in range(len(prompt_image_path)):
                base = prompt_image_path[i]
                r_i_list.append(np.load(base).tolist())
                base= Path(*[("text" if p == "image" else p) for p in Path(base).parts])
                r_t_list.append(np.load(base).tolist())

        elif (observed_text == 1) and (observed_image == 0):
            r_l_list = self.t2t_r_l_list_list[index]
            for i in range(len(prompt_text_path)):
                base = prompt_text_path[i]
                r_t_list.append(np.load(base).tolist())
                base= Path(*[("image" if p == "text" else p) for p in Path(base).parts])
                r_i_list.append(np.load(base).tolist())

        elif (observed_text == 1) and (observed_image == 1):
            r_l_list = self.i2i_r_l_list_list[index]
            for prompt_image,prompt_text in zip(prompt_image_path, prompt_text_path):
                r_i_list.append(np.load(prompt_image).tolist())
                r_t_list.append(np.load(prompt_text).tolist())
        else:
            raise ValueError(f"No available modalities for item: {index}")

        return {
            "image": image,
            "text": text,
            "label": label,
            "r_t_list": r_t_list,
            "r_i_list": r_i_list,
            "r_l_list": r_l_list,
            "observed_text": observed_text,
            "observed_image": observed_image
        }


    def __len__(self):
        return len(self.label_list)


class RAGPTCollator():


    def __init__(self, tokenizer: BertTokenizer = None, image_processor: ViltImageProcessor = None,
                 max_text_len: int = 128):

        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('dandelin/vilt-b32-mlm', do_lower_case=True)
        if image_processor is None:
            image_processor = ViltImageProcessor.from_pretrained('dandelin/vilt-b32-mlm')

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_len = max_text_len


    def __call__(self, batch):
        text = [item['text'] for item in batch]
        image = [item['image'] for item in batch]
        label = [item['label'] for item in batch]
        r_t_list = [item['r_t_list'] for item in batch]
        r_i_list = [item['r_i_list'] for item in batch]
        observed_text = [item['observed_text'] for item in batch]
        observed_image = [item['observed_image'] for item in batch]
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
            "r_l_list": r_l_list,
            "observed_image": torch.tensor(observed_image,dtype=torch.int64),
            "observed_text": torch.tensor(observed_text,dtype=torch.int64)
        }
