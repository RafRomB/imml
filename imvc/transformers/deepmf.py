import torch
import torch.nn
from torch.nn import init
from torch.nn.parameter import Parameter
from sklearn.decomposition import FastICA
import math
import numpy as np
import lightning.pytorch as pl


class DeepMFDataset(torch.utils.data.Dataset):

    def __init__(self, Xs, transform = None, target_transform = None):
        self.Xs = Xs
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.Xs)


    def __getitem__(self, idx):
        X = self.Xs
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


class DeepMF(pl.LightningModule):

    def __init__(self, X, K=10, n_layers:int =3, learning_rate:float =1e-2, alpha:float = 0.01,
                 neighbor_proximity='Lap', problem='regression'):
        super().__init__()
        self.M, self.N = X.shape
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.neighbor_proximity = neighbor_proximity
        self.problem = problem

        if self.problem == 'regression':
            self.loss_fun = torch.nn.MSELoss()
        else:
            self.loss_fun = torch.nn.BCELoss()

        layers = [_SparseLinear(self.M, K)]
        for i in range(n_layers):
            layers.append(torch.nn.Linear(K, K))
            if n_layers > 1:
                layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(K, self.N))
        if self.problem == 'classification':
            layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*layers)
        self.Su, self.Du = self._get_similarity(X, 'row')
        self.Sv, self.Dv = self._get_similarity(X, 'col')


    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0001)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        y_pred, local_y = self._clean_y(y_pred, y)
        loss = self.loss_fun(y_pred, local_y)
        u_loss, v_loss = self._U_V_loss(self.Su, self.Sv, self.Du, self.Dv)
        loss = (1 - self.alpha) * loss + self.alpha * (u_loss + v_loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        y_pred, local_y = self._clean_y(y_pred, y)
        loss = self.loss_fun(y_pred, local_y)
        u_loss, v_loss = self._U_V_loss(self.Su, self.Sv, self.Du, self.Dv)
        loss = (1 - self.alpha) * loss + self.alpha * (u_loss + v_loss)
        return loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        y_pred, local_y = self._clean_y(y_pred, y)
        loss = self.loss_fun(y_pred, local_y)
        u_loss, v_loss = self._U_V_loss(self.Su, self.Sv, self.Du, self.Dv)
        loss = (1 - self.alpha) * loss + self.alpha * (u_loss + v_loss)
        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, _ = batch
        pred = self.model(x)
        if self.problem == 'classification':
            pred[pred <= 0.5] = 0
            pred[pred > 0.5] = 1
        return pred


    def _get_similarity(self, data, row_or_col, n=15):
        self.data_isnan = torch.isnan(data)
        if torch.any(self.data_isnan):
            D = self._get_distance_with_nan(data, self.data_isnan, row_or_col)
        else:
            D = self._get_distance(data, row_or_col)
        S = 1 / (1 + D)
        N, _ = D.shape
        if not n:
            n = int(N / self.K)
        cutoff = D[range(N), torch.argsort(D, dim=1)[:, n-1]]
        Indicator = torch.zeros(N, N, dtype=torch.float)
        for i in range(N):
            for j in range(N):
                if D[i, j] <= cutoff[i] or D[i, j] <= cutoff[j]:
                    Indicator[i, j] = 1
        # sigma = D.std()/2
        # S = torch.exp(-D/(2*sigma*sigma))
        S = Indicator * S
        return S, D


    def _get_distance(self, data, row_or_col):
        M, N = data.shape
        if row_or_col == 'row':
            G = torch.mm(data, data.t())
        else:
            G = torch.mm(data.t(), data)
        g = torch.diag(G)
        if row_or_col == 'row':
            x = g.repeat(M, 1)
        else:
            x = g.repeat(N, 1)
        D = x + x.t() - 2*G
        return D


    def _get_distance_with_nan(self, o_data, data_isnan, row_or_col):
        data = o_data.clone().detach()
        data[torch.isnan(data)] = 0
        D = self._get_distance(data, row_or_col)
        M, N = data.shape
        if row_or_col == 'row':
            row_or_col_nan = torch.sum(data_isnan, dim=1).repeat(M, 1)
            penalty = torch.max(data) * 1.0 / N
        else:
            row_or_col_nan = torch.sum(data_isnan, dim=0).repeat(N, 1)
            penalty = torch.max(data) * 1.0 / M
        if penalty == 0:
            penalty = 1
        nan_penalty = penalty * (row_or_col_nan + row_or_col_nan.t())
        nan_penalty = nan_penalty.float()
        D = D + nan_penalty - torch.diag(torch.diag(nan_penalty))

        return D


    def _U_V_loss(self, Su, Sv, Du, Dv):
        U, V = self._load_U_V()

        if self.neighbor_proximity == 'Lap':
            u_loss = self._lap_loss(U, Su, 'row')
            v_loss = self._lap_loss(V, Sv, 'col')
        elif self.neighbor_proximity == 'MSE':
            u_loss = torch.sum(torch.pow(self._get_distance(U, 'row') - Du, 2)) / (self.M * self.M) * 0.01
            v_loss = torch.sum(torch.pow(self._get_distance(V, 'col') - Dv, 2)) / (self.N * self.N) * 0.01
        elif self.neighbor_proximity == 'KL':
            u_loss = self._kl_loss(1 / (1+Du), 1 / (1 + self._get_distance(U, 'row')))
            v_loss = self._kl_loss(1 / (1+Dv), 1 / (1 + self._get_distance(V, 'col')))
        else:
            u_loss = 0
            v_loss = 0

        return u_loss, v_loss

    def _kl_loss(self, P, Q):
        # return torch.sum(P*torch.log2(P/Q))

        return torch.log(torch.sum((P*P/Q)-P+Q))

    def _lap_loss(self, W, Sw, row_or_col):
        return torch.sum(self._get_distance(W, row_or_col) * Sw)

    def _initial_U_V(self, data):
        model = FastICA(n_components=self.K)

        U = model.fit_transform(np.nan_to_num(data))
        U = torch.tensor(U, dtype=torch.float)

        V = model.fit_transform(np.nan_to_num(data).T)
        V = torch.tensor(V, dtype=torch.float)

        self.model[0].weight.data = U.t()
        self.model[-1].weight.data = V

    def _load_U_V(self):
        U = self.model[0].weight.t()
        V = self.model[-1].weight.t()
        return U, V

    def _save_U_V(self):
        U, V = self._load_U_V()
        U = U.data.cpu().detach().numpy()
        V = V.data.cpu().detach().numpy()
        return U, V

    def _clean_y(self, y_pred, y):
        y_pred[torch.isnan(y)] = 0
        y[torch.isnan(y)] = 0
        return y_pred, y


class _SparseLinear(torch.nn.Module):

    _constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(_SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):

        if input.dim() == 2 and self.bias is not None:
            # fused op is marginally faster
            ret = torch.sparse.addmm(self.bias, input, self.weight.t())
        else:
            output = torch.sparse.mm(input, self.weight.t())
            if self.bias is not None:
                output += self.bias
            ret = output
        return ret

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
