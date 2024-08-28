try:
    import torch
    import lightning as pl
    torch_installed = True
except ImportError:
    torch_installed = False
    error_message = "torch and lightning needs to be installed."


class MRGCNDataset(torch.utils.data.Dataset):

    def __init__(self, Xs, transform = None):
        self.Xs = Xs
        self.transform = transform


    def __len__(self):
        return len(self.Xs[0])


    def __getitem__(self, idx):
        if self.transform is not None:
            Xs = [self.transform[X_idx](X[idx]) for X_idx ,X in enumerate(self.Xs)]
        else:
            Xs = [X[idx] for X in self.Xs]
        Xs = tuple(Xs)
        return Xs


class DeepMFDataset(torch.utils.data.Dataset):

    def __init__(self, X, transform = None, target_transform = None):
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
