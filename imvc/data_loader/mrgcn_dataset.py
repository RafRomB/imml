try:
    import torch
    import lightning as pl
    torch_installed = True
except ImportError:
    torch_installed = False
    torch_module_error = "torch and lightning needs to be installed."


class MRGCNDataset(torch.utils.data.Dataset):

    def __init__(self, Xs: list, transform = None):
        if not torch_installed:
            raise ImportError(torch_module_error)
        if not isinstance(Xs, list):
            raise ValueError(f"Invalid Xs. It must be a list of array-likes. A {type(Xs)} was passed.")

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