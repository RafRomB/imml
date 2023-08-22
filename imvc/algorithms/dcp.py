import copy
import itertools
import torch
from torch import nn
import torch.nn.functional as F


class DCP():
    # Dual contrastive prediction for multi-view
    def __init__(self, autoencoders_units:list, predictions_units:list, learning_rate:float = 3.0e-4, approach:str = "CG",
                 start_dual_prediction:int = 50, alpha:int = 10, lambda2:float = 0.1, lambda1:float = 0.1,
                 activation:str = 'relu', batchnorm:bool = True):
        """Constructor.

        Args:
            config: parameters defined in configure.py.
        """
        approach_types = ['CG', 'CV']
        if approach not in approach_types:
            raise ValueError(f"Invalid value for approach. Expected one of: {approach_types}")

        self.autoencoders_units = autoencoders_units
        self.predictions_units = predictions_units
        self.learning_rate = learning_rate
        self.approach = approach
        self.start_dual_prediction = start_dual_prediction
        self.alpha = alpha
        self.lambda2 = lambda2
        self.lambda1 = lambda1
        self.activation = activation
        self.batchnorm = batchnorm

        self._latent_dim_ = autoencoders_units[0][-1]
        self._dims_view_, self.autoencoders_, self.combinations_ = [], [], []

        for autoencoder_units, prediction_units in zip(autoencoders_units, predictions_units):
            dims_view = [self._latent_dim_] + prediction_units
            autoencoder = Autoencoder(encoder_dim = autoencoder_units, activation=activation, batchnorm=batchnorm)
            self._dims_view_.append(dims_view)
            self.autoencoders_.append(autoencoder)

        for subset in itertools.combinations(list(range(len(self._dims_view_))), 2):
            predictions0 = Prediction(prediction_dim = self._dims_view_[subset[0]], activation=activation,
                                      batchnorm=batchnorm)
            predictions1 = Prediction(prediction_dim = self._dims_view_[subset[1]], activation=activation,
                                      batchnorm=batchnorm)
            self.combinations_.append(predictions0)
            self.combinations_.append(predictions1)


    def configure_optimizers(self):
        models = [model.parameters() for model in [*self.autoencoders_, *self.combinations_]]
        return torch.optim.Adam(itertools.chain(*models), lr=self.learning_rate)


    def training_step(self, batch, batch_idx):

        z_halfs, reconstruction_loss = [], 0
        for autoencoder,view_data in zip(self.autoencoders_, batch):
            # get the hidden states for each view
            z_half = autoencoder.encoder(view_data)
            # Within-view Reconstruction Loss
            reconstruction_loss += F.mse_loss(autoencoder.decoder(z_half), view_data)
            z_halfs.append(z_half)

        z_half1, losses_icl, dualprediction_losses = z_halfs[0], [], []
        for idx, z_half in enumerate(z_halfs[1:]):
            # Instance-level Contrastive Loss
            loss_icl = instance_contrastive_Loss(x_out = z_half1, x_tf_out = z_half, lamb = self.alpha)
            losses_icl.append(loss_icl)
            # Cross-view Dual-Prediction Loss
            dualprediction_loss, _ = self.combinations_[idx*2](z_half1)
            pre = F.mse_loss(dualprediction_loss, z_half)
            dualprediction_losses.append(pre)
            dualprediction_loss, _ = self.combinations_[idx*2 +1](z_half)
            pre = F.mse_loss(dualprediction_loss, z_half1)
            dualprediction_losses.append(pre)

        n = len(dualprediction_losses)
        if self.approach == "CG":
            for z_half_subset in itertools.combinations(z_halfs[1:]):
                # Instance-level Contrastive Loss
                loss_icl = instance_contrastive_Loss(x_out=z_half_subset[0], x_tf_out=z_half_subset[1], lamb=self.alpha)
                losses_icl.append(loss_icl)
                # Cross-view Dual-Prediction Loss
                dualprediction_loss, _ = self.combinations_[idx*2 + n](z_half_subset[0])
                pre = F.mse_loss(dualprediction_loss, z_half_subset[1])
                dualprediction_losses.append(pre)
                dualprediction_loss, _ = self.combinations_[idx*2 +1 + n](z_half_subset[1])
                pre = F.mse_loss(dualprediction_loss, z_half_subset[0])
                dualprediction_losses.append(pre)

        losses_icl = losses_icl[0] + [0.1*loss_icl for loss_icl in losses_icl]
        losses_icl = torch.mean(losses_icl)
        dualprediction_losses = torch.mean(dualprediction_losses) *2
        all_loss = losses_icl + reconstruction_loss * self.lambda2
        if batch_idx >= self.start_dual_prediction:
            all_loss += self.lambda1 * dualprediction_losses

        return all_loss


    def validation_step(self, batch, batch_idx):
        z_halfs, reconstruction_loss = [], 0
        for autoencoder,view_data in zip(self.autoencoders_, batch):
            # get the hidden states for each view
            z_half = autoencoder.encoder(view_data)
            # Within-view Reconstruction Loss
            reconstruction_loss += F.mse_loss(autoencoder.decoder(z_half), view_data)
            z_halfs.append(z_half)

        z_half1, losses_icl, dualprediction_losses = z_halfs[0], [], []
        for idx, z_half in enumerate(z_halfs[1:]):
            # Instance-level Contrastive Loss
            loss_icl = instance_contrastive_Loss(x_out = z_half1, x_tf_out = z_half, lamb = self.alpha)
            losses_icl.append(loss_icl)
            # Cross-view Dual-Prediction Loss
            dualprediction_loss, _ = self.combinations_[idx*2](z_half1)
            pre = F.mse_loss(dualprediction_loss, z_half)
            dualprediction_losses.append(pre)
            dualprediction_loss, _ = self.combinations_[idx*2 +1](z_half)
            pre = F.mse_loss(dualprediction_loss, z_half1)
            dualprediction_losses.append(pre)

        n = len(dualprediction_losses)
        if self.approach == "CG":
            for z_half_subset in itertools.combinations(z_halfs[1:]):
                # Instance-level Contrastive Loss
                loss_icl = instance_contrastive_Loss(x_out=z_half_subset[0], x_tf_out=z_half_subset[1], lamb=self.alpha)
                losses_icl.append(loss_icl)
                # Cross-view Dual-Prediction Loss
                dualprediction_loss, _ = self.combinations_[idx*2 + n](z_half_subset[0])
                pre = F.mse_loss(dualprediction_loss, z_half_subset[1])
                dualprediction_losses.append(pre)
                dualprediction_loss, _ = self.combinations_[idx*2 +1 + n](z_half_subset[1])
                pre = F.mse_loss(dualprediction_loss, z_half_subset[0])
                dualprediction_losses.append(pre)

        losses_icl = losses_icl[0] + [0.1*loss_icl for loss_icl in losses_icl]
        losses_icl = torch.mean(losses_icl)
        dualprediction_losses = torch.mean(dualprediction_losses) *2
        all_loss = losses_icl + reconstruction_loss * self.lambda2
        if batch_idx >= self.start_dual_prediction:
            all_loss += self.lambda1 * dualprediction_losses

        return all_loss


    def test_step(self, batch, batch_idx):
        z_halfs, reconstruction_loss = [], 0
        for autoencoder,view_data in zip(self.autoencoders_, batch):
            # get the hidden states for each view
            z_half = autoencoder.encoder(view_data)
            # Within-view Reconstruction Loss
            reconstruction_loss += F.mse_loss(autoencoder.decoder(z_half), view_data)
            z_halfs.append(z_half)

        z_half1, losses_icl, dualprediction_losses = z_halfs[0], [], []
        for idx, z_half in enumerate(z_halfs[1:]):
            # Instance-level Contrastive Loss
            loss_icl = instance_contrastive_Loss(x_out = z_half1, x_tf_out = z_half, lamb = self.alpha)
            losses_icl.append(loss_icl)
            # Cross-view Dual-Prediction Loss
            dualprediction_loss, _ = self.combinations_[idx*2](z_half1)
            pre = F.mse_loss(dualprediction_loss, z_half)
            dualprediction_losses.append(pre)
            dualprediction_loss, _ = self.combinations_[idx*2 +1](z_half)
            pre = F.mse_loss(dualprediction_loss, z_half1)
            dualprediction_losses.append(pre)

        n = len(dualprediction_losses)
        if self.approach == "CG":
            for z_half_subset in itertools.combinations(z_halfs[1:]):
                # Instance-level Contrastive Loss
                loss_icl = instance_contrastive_Loss(x_out=z_half_subset[0], x_tf_out=z_half_subset[1], lamb=self.alpha)
                losses_icl.append(loss_icl)
                # Cross-view Dual-Prediction Loss
                dualprediction_loss, _ = self.combinations_[idx*2 + n](z_half_subset[0])
                pre = F.mse_loss(dualprediction_loss, z_half_subset[1])
                dualprediction_losses.append(pre)
                dualprediction_loss, _ = self.combinations_[idx*2 +1 + n](z_half_subset[1])
                pre = F.mse_loss(dualprediction_loss, z_half_subset[0])
                dualprediction_losses.append(pre)

        losses_icl = losses_icl[0] + [0.1*loss_icl for loss_icl in losses_icl]
        losses_icl = torch.mean(losses_icl)
        dualprediction_losses = torch.mean(dualprediction_losses) *2
        all_loss = losses_icl + reconstruction_loss * self.lambda2
        if batch_idx >= self.start_dual_prediction:
            all_loss += self.lambda1 * dualprediction_losses

        return all_loss


    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        latent_evals, latent_code_evals, idx_evals, missing_idx_evals = [], [], [], []
        for autoencoder,autoencoder_units, view_data in zip(self.autoencoders_, self.autoencoders_units, batch):
            idx_eval = torch.isnan(view_data).sum(0) != view_data.shape[1]
            idx_evals.append(idx_eval)
            missing_idx_eval = ~idx_eval
            missing_idx_evals.append(missing_idx_eval)
            latent_eval = autoencoder.encoder(view_data[idx_eval])
            latent_evals.append(latent_eval)
            latent_code_eval = torch.zeros(view_data.shape[0], autoencoder_units[-1])
            latent_code_evals.append(latent_code_eval)

        first_missing_idx_eval = missing_idx_evals[0]
        if first_missing_idx_eval.sum() != 0:
            for idx, autoencoder, view_data, idx_eval in enumerate(zip(self.autoencoders_[1:], batch[1:], idx_evals[1:])):
                missing_idxs = [~i for i in idx_evals]
                missing_idxs[idx] = idx_evals[idx]
                onlyhas_idx = first_missing_idx_eval * torch.matmul(*missing_idxs)
                onlyhas = autoencoder.encoder(view_data[onlyhas_idx])
                onlyhas, _ = self.combinations_[idx*2+1](onlyhas)
                latent_code_evals[0][onlyhas_idx] = onlyhas
            missing_idxs = [i for i in idx_evals[1:]]
            onlyhas_idx = first_missing_idx_eval * torch.matmul(*missing_idxs)
            onlyhas =  [self.combinations_[idx*2+1](autoencoder.encoder(view_data[onlyhas_idx]))[0]
                        for autoencoder, view_data in zip(self.autoencoders_[1:], batch[1:])]
            onlyhas = torch.stack(onlyhas).mean(0)
            latent_evals[0][onlyhas_idx] = onlyhas

        for idx, missing_idx_eval in enumerate(missing_idx_evals[1:], start=1):
            if missing_idx_eval.sum() != 0:
                has_idx = missing_idx_eval * idx_evals[0]
                has = self.autoencoders_[0](batch[0][has_idx])
                has,_ = self.combinations_[idx*2](has)
                latent_code_evals[idx][has_idx] = has

                missing_idxs = [i for id,i in enumerate(idx_evals) if id not in [0, idx]]
                onlyhas_idx = missing_idx_eval * torch.matmul(*missing_idxs) * ~idx_evals[0]
                for autoencoder in zip(self.autoencoders_, view_data)







            ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
            ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
            ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

            onlyhas = self.autoencoder2.encoder(x2_train[ano_bonlyhas_idx])
            onlyhas, _ = self.b2a(onlyhas)

            ano_conlyhas = self.autoencoder3.encoder(x3_train[ano_conlyhas_idx])
            ano_conlyhas, _ = self.c2a(ano_conlyhas)

            ano_bcbothhas_1 = self.autoencoder2.encoder(x2_train[ano_bcbothhas_idx])
            ano_bcbothhas_2 = self.autoencoder3.encoder(x3_train[ano_bcbothhas_idx])
            ano_bcbothhas = (self.b2a(ano_bcbothhas_1)[0] + self.c2a(ano_bcbothhas_2)[0]) / 2.0

            latent_code_a_eval[ano_bonlyhas_idx] = onlyhas
            latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
            latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas



        return pred


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent



def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def instance_contrastive_Loss(x_out, x_tf_out, EPS, lamb=1.0):
    """Contrastive loss for maximizng the consistency"""
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss


def category_contrastive_loss(repre, gt, classes, flag_gt):
    """Category-level contrastive loss.

    This function computes loss on the representation corresponding to its groundtruth (repre, gt).  A

    Args:
      repre: [N, D] float tensor.
      gt: [N, 1] float tensor.
      classes:  int tensor.

    Returns:
      loss:  float tensor.
    """

    if flag_gt == True:
        gt = gt - 1

    batch_size = gt.size()[0]
    F_h_h = torch.matmul(repre, repre.t())
    F_hn_hn = torch.diag(F_h_h)
    F_h_h = F_h_h - torch.diag_embed(F_hn_hn)

    label_onehot = torch.nn.functional.one_hot(gt, classes).float()

    label_num = torch.sum(label_onehot, 0, keepdim=True)
    F_h_h_sum = torch.matmul(F_h_h, label_onehot)
    label_num_broadcast = label_num.repeat([gt.size()[0], 1]) - label_onehot
    label_num_broadcast[label_num_broadcast == 0] = 1
    F_h_h_mean = torch.div(F_h_h_sum, label_num_broadcast)
    gt_ = torch.argmax(F_h_h_mean, dim=1)  # gt begin from 0
    F_h_h_mean_max = torch.max(F_h_h_mean, dim=1)[0]
    theta = (gt == gt_).float()
    F_h_hn_mean_ = F_h_h_mean.mul(label_onehot)
    F_h_hn_mean = torch.sum(F_h_hn_mean_, dim=1)
    return torch.sum(torch.relu(torch.add(theta, torch.sub(F_h_h_mean_max, F_h_hn_mean))))