try:
    import torch
    torch_installed = True
except ImportError:
    torch_installed = False
    torch_module_error = "torch needs to be installed."


class DeepMFDataset(torch.utils.data.Dataset):

    def __init__(self, X, transform = None, target_transform = None):
        if not torch_installed:
            raise ImportError(torch_module_error)
        if not isinstance(X, torch.Tensor):
            raise ValueError(f"Invalid X. It must be a tensor. A {type(X)} was passed.")

        self.X = X
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        X = self.X
        label = X[idx, :]
        Xi = torch.tensor([idx], dtype=torch.long).unsqueeze(0)
        Xv = torch.ones(1, dtype=torch.float)
        X = torch.sparse_coo_tensor(Xi, Xv, (len(X),))
        label = label.to(X.dtype)

        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            label = self.target_transform(label)
        return X, label
