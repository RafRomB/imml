import os
import numpy as np
import pandas as pd
from PIL import Image

from ..classify._ragpt.vilt import ViltModel, ViltImageProcessor

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import AutoModel, AutoProcessor, BertTokenizer

    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'Deep' needs to be installed."

nnModuleBase = nn.Module if deepmodule_installed else object


class MCR(nnModuleBase):
    r"""
    NEighborhood based Multi-Omics clustering (NEMO).

    NEMO is a method used for clustering data from multiple modalities sources. This algorithm operates
    through three main stages. Initially, it constructs a similarity matrix for each modality that represents the
    similarities between different samples. Then, it merges these individual modality matrices into a unified one,
    combining the information from all modalities. Finally, the algorithm performs the actual clustering process on this
    integrated network, grouping similar samples together based on their multi-modal data patterns.

    Parameters
    ----------
    n_clusters : int or list-of-int
        The number of clusters to generate. If it is a list, the number of clusters will be estimated by the algorithm
        with this range of number of clusters to choose between.
    num_neighbors : list or int, default=None
        The number of neighbors to use for each modality. It can either be a number, a list of numbers or None. If it is a
        number, this is the number of neighbors used for all modalities. If this is a list, the number of neighbors are
        taken for each modality from that list. If it is None, each modality chooses the number of neighbors to be the number
        of samples divided by num_neighbors_ratio.
    num_neighbors_ratio : int, default=6
        The number of clusters to generate. If it is not provided, it will be estimated by the algorithm.
    metric : str or list-of-str, default="sqeuclidean"
        Distance metric to compute. Must be one of available metrics in :py:func`scipy.spatial.distance.pdist`. If
        multiple arrays a provided an equal number of metrics may be supplied.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default='python'
        Engine to use for computing the model. Must be one of ["python", "r"].
    verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    embedding_ : array-like of shape (n_samples, n_clusters)
        The final representation of the data to be used as input for the clustering step.
    n_clusters_ : int
        Final number of clusters.
    num_neighbors_ : int
        Final number of neighbors.
    affinity_matrix_ : np.array (n_samples, n_samples)
        Affinity matrix.

    References
    ----------
    .. [#nemopaper] Rappoport Nimrod, Shamir Ron. NEMO: Cancer subtyping by integration of partial multi-omic data.
                    Bioinformatics. 2019;35(18):3348–3356. doi: 10.1093/bioinformatics/btz058.
    .. [#nemocode] https://github.com/Shamir-Lab/NEMO

    Example
    --------
    >>> from imml.retrieve import MCR
    >>> images = ["1.png", "2.png", "3.png", "4.png"] # image paths
    >>> texts = ["I like reading.", "I like pizza.", "I drink watter.", "I went to the cinema."] # image paths
    >>> Xs = [images, texts]
    >>> y = [0, 0, 1, 1]
    >>> modalities = ["image", "text"]
    >>> estimator = MCR(modalities=modalities)
    >>> estimator.fit(Xs=Xs, y=y)
    >>> preds = estimator.predict(Xs=Xs)
    >>> memory_bank = estimator.transform(Xs=Xs)
    >>> estimator.generate_cap(path="", database= database)
    """


    def __init__(self, batch_size: int = 64, n_neighbors: int = 20, device: str = "cpu",
                 modalities: list = None, pretrained_model: AutoModel = None, processor: AutoProcessor = None,
                 generate_cap: bool = False, prompt_path: str = None, pretrained_vilt: ViltModel = None,
                 tokenizer: BertTokenizer = None, image_processor: ViltImageProcessor = None,
                 max_text_len: int = 128, max_image_len: int = 145, save_memory_bank: bool = True):

        if modalities is None:
            raise ValueError(f"Invalid modalities. It must be a list. A {type(modalities)} was passed.")

        super().__init__()

        self.modalities = modalities
        self.batch_size = batch_size
        self.n_neighbors = n_neighbors
        self.device = device
        self.save_memory_bank = save_memory_bank
        if pretrained_model is None:
            self.pretrained_model = AutoModel.from_pretrained("openai/clip-vit-large-patch14-336").to(self.device)
        if processor is None:
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        self.generate_cap = generate_cap
        if generate_cap:
            self.prompt_path = prompt_path
            self.pretrained_vilt = pretrained_vilt
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self.max_text_len = max_text_len
            self.max_image_len = max_image_len


    def fit(self, Xs: list, y):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self :  Fitted estimator.
        """

        q_i_list, q_t_list = self._encode_img_text(Xs=Xs)
        memory_bank = pd.DataFrame({
            'item_id': list(range(len(Xs[0]))),
            'img_path': Xs[self.modalities.index("image")],
            'text': Xs[self.modalities.index("text")],
            'q_i': q_i_list,
            'q_t': q_t_list,
            'label': y,
        })
        memory_bank = memory_bank.reset_index(drop=True)
        if self.generate_cap:
            self._cap(prompt_path=self.prompt_path, pretrained_vilt=self.pretrained_vilt, memory_bank=memory_bank,
                     tokenizer=self.tokenizer, image_processor=self.image_processor,
                     max_text_len=self.max_text_len, max_image_len=self.max_image_len)

        if isinstance(y, pd.Series):
            memory_bank["item_id"] = y.index
            memory_bank.index = y.index
        if self.save_memory_bank:
            self.memory_bank = memory_bank
            output = self
        else:
            output = memory_bank

        return output


    def predict(self, Xs: list = None, memory_bank: pd.DataFrame = None, n_neighbors: int = None):

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if memory_bank is None:
            memory_bank = self.memory_bank
        if Xs is not None:
            q_i, q_t = self._encode_img_text(Xs=Xs)
        else:
            q_i = memory_bank['q_i'].tolist()
            q_t = memory_bank['q_t'].tolist()

        r_v_i = memory_bank['q_i'].tolist()
        r_v_t = memory_bank['q_t'].tolist()
        memory_bank_id = memory_bank['item_id'].tolist()
        memory_bank_label = memory_bank['label'].tolist()

        q_i = torch.tensor(q_i).squeeze(1).to(self.device)
        q_t = torch.tensor(q_t).squeeze(1).to(self.device)
        r_v_i = torch.tensor(r_v_i).squeeze(1).to(self.device)
        r_v_t = torch.tensor(r_v_t).squeeze(1).to(self.device)

        t2t_id_list, t2t_sims_list, t2t_label_list = \
            self._compute_similarity_in_batches(q_t ,r_v_t, memory_bank_id, memory_bank_label, n_neighbors)
        i2i_id_list, i2i_sims_list, i2i_label_list = \
            self._compute_similarity_in_batches(q_i ,r_v_i, memory_bank_id, memory_bank_label, n_neighbors)
        output = {
            "image": {
                "id": i2i_id_list,
                "sims": i2i_sims_list,
                "label": i2i_label_list,
            },
            "text": {
                "id": t2t_id_list,
                "sims": t2t_sims_list,
                "label": t2t_label_list,
            },
        }

        return output


    def fit_predict(self, Xs: list, y: list, save_memory_bank: bool = True, n_neighbors: int = None):
        if save_memory_bank:
            memory_bank = self.fit(Xs=Xs, y=y)
        else:
            memory_bank = self.fit(Xs=Xs, y=y).memory_bank

        output = self.predict(memory_bank=memory_bank, n_neighbors=n_neighbors)
        return output


    def transform(self, Xs: list, y: list, memory_bank: pd.DataFrame = None, n_neighbors: int = None, save_memory_bank: bool = True):
        if memory_bank is None:
            memory_bank = self.memory_bank
        output = self.predict(Xs=Xs, memory_bank=memory_bank, n_neighbors=n_neighbors)
        database = pd.DataFrame({
            'item_id': list(range(len(Xs[0]))),
            'img_path': Xs[self.modalities.index("image")],
            'text': Xs[self.modalities.index("text")],
            'label': y,
        })
        if isinstance(y, pd.Series):
            database["item_id"] = y.index
            database.index = y.index
        observed_mod_indicator = pd.DataFrame({f"observed_{mod}": pd.notna(X).tolist()
                                               for X, mod in zip(Xs, self.modalities)}, index=database.index)
        database = pd.concat([database, observed_mod_indicator.astype(int)], axis=1)
        for mod in self.modalities:
            key = mod[0]
            key = f"{key}2{key}"
            for obj in list(output[mod].keys()):
                final_key = f"{key}_{obj}_list"
                database[final_key] = output[mod][obj]
            database[f"prompt_{mod}_path"] = [memory_bank.loc[id, f"prompt_{mod}_path"].to_list() for id in output[mod]["id"]]
        return database


    def fit_transform(self, Xs: list, y, save_memory_bank: bool = True, n_neighbors: int = None):
        if save_memory_bank:
            memory_bank = self.fit(Xs=Xs, y=y, save_memory_bank=save_memory_bank).memory_bank
        else:
            memory_bank = self.fit(Xs=Xs, y=y, save_memory_bank=save_memory_bank)

        database = self.transform(memory_bank=memory_bank, n_neighbors=n_neighbors)
        return database


    def _cap(self, prompt_path: str, memory_bank: pd.DataFrame = None, pretrained_vilt: ViltModel = None,
            tokenizer: BertTokenizer = None, image_processor: ViltImageProcessor = None,
            max_text_len: int = 128, max_image_len: int = 145):

        if pretrained_vilt is None:
            pretrained_vilt = ViltModel.from_pretrained('dandelin/vilt-b32-mlm')
        self.embedding_layer = pretrained_vilt.embeddings
        for param in self.embedding_layer.parameters():
            param.requires_grad = False
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('dandelin/vilt-b32-mlm', do_lower_case=True)
        if image_processor is None:
            image_processor = ViltImageProcessor.from_pretrained('dandelin/vilt-b32-mlm')

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_len = max_text_len
        self.max_image_len = max_image_len

        if memory_bank is None:
            memory_bank = self.memory_bank
        n_chunks = -(-len(memory_bank) // self.batch_size)
        for chunk in np.array_split(memory_bank, n_chunks):
            texts = chunk['text']
            ids = chunk['img_path']

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
            for image_path in ids:
                image = Image.open(image_path).convert("RGB") \
                    if pd.notna(image_path) else Image.new("RGB", (256, 256), (0, 0, 0))
                image = self._resize_image(image)
                images.append(image)

            encoding_image_processor = self.image_processor(images, return_tensors="pt")
            pixel_values = encoding_image_processor["pixel_values"]
            pixel_mask = encoding_image_processor["pixel_mask"]

            emb = self._encode(input_ids, pixel_values, pixel_mask, token_type_ids, attention_mask)
            text_emb = emb[:, :self.max_text_len]
            image_emb = emb[:, self.max_text_len:]

            os.makedirs(os.path.join(prompt_path, "text"), exist_ok=True)
            os.makedirs(os.path.join(prompt_path, "image"), exist_ok=True)

            for i, id in enumerate(ids):
                idx = chunk.iloc[i].name
                file_name = os.path.basename(id).split(".")[0]
                file_path = os.path.join(prompt_path, "image", f"{file_name}.npy")
                memory_bank.loc[idx, "prompt_image_path"] = file_path
                np.save(file_path, image_emb[i].detach().numpy())
                file_path = os.path.join(prompt_path, "text", f"{file_name}.npy")
                memory_bank.loc[idx, "prompt_text_path"] = file_path
                np.save(file_path, text_emb[i].detach().numpy())


    def _encode_img_text(self, Xs: list):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self :  Fitted estimator.
        """

        q_i_list = []
        q_t_list = []
        for X,mod in zip(Xs, self.modalities):
            if mod == "image":
                if any([isinstance(file, str) for file in X]):
                    all_images = [Image.open(img_name)
                                  if pd.notna(img_name) else Image.new("RGBA", (256, 256), (0, 0, 0))
                                  for img_name in X]
                else:
                    all_images = [img
                                  if pd.notna(img) else Image.new("RGBA", (256, 256), (0, 0, 0))
                                  for img in X]
                batch_size = self.batch_size
                for i in range(0, len(all_images), batch_size):
                    batch_images = all_images[i: i + batch_size]
                    batch_outputs = self._encode_image(batch_images)
                    q_i_list.extend(batch_outputs.cpu().tolist())
                q_i_list = [q_i if pd.notna(img) else [torch.nan for q in q_i] for img,q_i in zip(X, q_i_list)]
            elif mod == "text":
                q_t_list = [self._encode_text(text).tolist()
                            if pd.notna(text) else self._encode_text("").tolist()
                            for text in X]
                q_t_list = [q_t if pd.notna(text) else [torch.nan for q in q_t] for text,q_t in zip(X, q_t_list)]
            else:
                raise ValueError(f"Unknown modality type: {mod}")

        return q_i_list, q_t_list


    def _compute_similarity_in_batches(self, query_vectors, memory_bank, memory_bank_id, memory_bank_label, n_neighbors):
        r_id_list = []
        sims_list = []
        r_label_list = []
        for i in range(0, len(query_vectors), self.batch_size):
            batch = query_vectors[i: i +self.batch_size].unsqueeze(1)
            similarity = F.cosine_similarity(batch, memory_bank.unsqueeze(0), dim=-1)
            sim_scores, top_k_id = torch.topk(similarity, k=n_neighbors+1, dim=-1)
            for j in range(batch.size(0)):
                id_index = i + j
                if torch.isnan(sim_scores[j]).all():
                    retrieved_ids = []
                    sim_score = []
                    retrieved_labels = []
                else:
                    id = memory_bank_id[id_index] if id_index < len(memory_bank_id) else None
                    retrieved_ids = [memory_bank_id[idx] for idx in top_k_id[j].tolist() if memory_bank_id[idx] != id]
                    retrieved_labels = [memory_bank_label[idx] for idx in top_k_id[j].tolist() if memory_bank_id[idx] != id]
                    sim_score = sim_scores[j ,1:].tolist()
                    if len(retrieved_ids) > n_neighbors:
                        retrieved_ids = retrieved_ids[:n_neighbors]
                        sim_score = sim_score[:n_neighbors]
                        retrieved_labels = retrieved_labels[:n_neighbors]
                r_id_list.append(retrieved_ids)
                sims_list.append(sim_score)
                r_label_list.append(retrieved_labels)
        return  r_id_list ,sims_list, r_label_list


    def _encode_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True ,truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        text_features = self.pretrained_model.text_model.embeddings(input_ids, attention_mask)
        text_features = self.pretrained_model.text_model.encoder(text_features).last_hidden_state
        text_features = self.pretrained_model.text_model.final_layer_norm(text_features)
        text_features = (self.pretrained_model.text_projection(text_features))
        return text_features[0, -1, :]


    def _encode_image(self, images):
        with torch.no_grad():
            processed_images = self.processor(images=images, return_tensors="pt").to(self.device)
            image_features = self.pretrained_model.vision_model.embeddings(processed_images['pixel_values'])
            image_features = self.pretrained_model.vision_model.pre_layrnorm(image_features)
            image_features = self.pretrained_model.vision_model.encoder(image_features).last_hidden_state
            image_features = self.pretrained_model.vision_model.post_layernorm(image_features)
            image_features = self.pretrained_model.visual_projection(image_features)
            return image_features[:, 0, :]


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

