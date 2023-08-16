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

        z_half1, loss_icl = z_halfs[0], []
        for z_half in z_halfs[1:]:
            # Instance-level Contrastive Loss
            loss_icli = instance_contrastive_Loss(x_out = z_half1, x_tf_out = z_half, lamb = self.alpha)
            loss_icl.append(loss_icli)

        if self.approach == "CG":

        loss_icl = loss_icl[0] + [0.1*i for i in loss_icl]
        loss_icl = torch.mean(loss_icl)




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



    def train_completegraph(self, config, logger, accumulated_metrics, x1_train, x2_train, x3_train, Y_list, mask,
                            optimizer, device):
        """Training the model with complete graph for clustering

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              x1_train: data of view 1
              x2_train: data of view 2
              x3_train: data of view 3
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari

        """

        epochs_total = config['training']['epoch']
        batch_size = config['training']['batch_size']

        # select the complete samples
        flag = torch.LongTensor([1, 1, 1]).to(device)
        flag = (mask == flag).int()
        flag = ((flag[:, 1] + flag[:, 0] + flag[:, 2]) == 3)
        train_view1 = x1_train[flag]
        train_view2 = x2_train[flag]
        train_view3 = x3_train[flag]

        for k in range(epochs_total):
            X1, X2, X3 = shuffle(train_view1, train_view2, train_view3)
            all0, all1, all2, all_icl, map1, map2 = 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, batch_x3, batch_No in next_batch_3view(X1, X2, X3, batch_size):
                # get the hidden states for each view
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)
                z_half3 = self.autoencoder3.encoder(batch_x3)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
                recon3 = F.mse_loss(self.autoencoder3.decoder(z_half3), batch_x3)

                reconstruction_loss = recon1 + recon2 + recon3

                # Instance-level Contrastive_Loss
                loss_icl1 = instance_contrastive_Loss(z_half1, z_half2, config['training']['alpha'])
                loss_icl2 = instance_contrastive_Loss(z_half1, z_half3, config['training']['alpha'])
                loss_icl3 = instance_contrastive_Loss(z_half2, z_half3, config['training']['alpha'])
                loss_icl = (loss_icl1 + 0.1 * loss_icl2 + 0.1 * loss_icl3) / 3

                # Cross-view Dual-Prediction Loss
                a2b, _ = self.a2b(z_half1)
                b2a, _ = self.b2a(z_half2)
                a2c, _ = self.a2c(z_half1)
                c2a, _ = self.c2a(z_half3)
                b2c, _ = self.b2c(z_half2)
                c2b, _ = self.c2b(z_half3)

                pre1 = F.mse_loss(a2b, z_half2)
                pre2 = F.mse_loss(b2a, z_half1)
                pre3 = F.mse_loss(a2c, z_half3)
                pre4 = F.mse_loss(c2a, z_half1)
                pre5 = F.mse_loss(b2c, z_half3)
                pre6 = F.mse_loss(c2b, z_half2)
                dualprediction_loss = (pre1 + pre2 + pre3 + pre4 + pre5 + pre6) / 3

                all_loss = loss_icl + reconstruction_loss * config['training']['lambda2']

                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda1'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                all0 += all_loss.item()
                all1 += recon1.item()
                all2 += recon2.item()
                map1 += pre1.item()
                map2 += pre2.item()
                all_icl += loss_icl.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Map loss = {:.4f} ===> Map loss = {:.4f} ===> loss_icl = {:.4e} ===> all loss = {:.4e}" \
                .format((k + 1), epochs_total, all1, all2, map1, map2, all_icl, all0)
            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval(), self.autoencoder3.eval()
                    self.a2b.eval(), self.b2a.eval()
                    self.b2c.eval(), self.c2b.eval()
                    self.a2c.eval(), self.c2a.eval()

                    # get the missing index
                    a_idx_eval = mask[:, 0] == 1
                    b_idx_eval = mask[:, 1] == 1
                    c_idx_eval = mask[:, 2] == 1
                    a_missing_idx_eval = mask[:, 0] == 0
                    b_missing_idx_eval = mask[:, 1] == 0
                    c_missing_idx_eval = mask[:, 2] == 0

                    a_latent_eval = self.autoencoder1.encoder(x1_train[a_idx_eval])
                    b_latent_eval = self.autoencoder2.encoder(x2_train[b_idx_eval])
                    c_latent_eval = self.autoencoder3.encoder(x3_train[c_idx_eval])

                    latent_code_a_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_b_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)
                    latent_code_c_eval = torch.zeros(x3_train.shape[0], config['Autoencoder']['arch3'][-1]).to(
                        device)

                    # for view a
                    if a_missing_idx_eval.sum() != 0:
                        ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
                        ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
                        ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

                        ano_bonlyhas = self.autoencoder2.encoder(x2_train[ano_bonlyhas_idx])
                        ano_bonlyhas, _ = self.b2a(ano_bonlyhas)

                        ano_conlyhas = self.autoencoder3.encoder(x3_train[ano_conlyhas_idx])
                        ano_conlyhas, _ = self.c2a(ano_conlyhas)

                        ano_bcbothhas_1 = self.autoencoder2.encoder(x2_train[ano_bcbothhas_idx])
                        ano_bcbothhas_2 = self.autoencoder3.encoder(x3_train[ano_bcbothhas_idx])
                        ano_bcbothhas = (self.b2a(ano_bcbothhas_1)[0] + self.c2a(ano_bcbothhas_2)[0]) / 2.0

                        latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
                        latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
                        latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas

                    # for view b
                    if b_missing_idx_eval.sum() != 0:
                        bno_aonlyhas_idx = b_missing_idx_eval * a_idx_eval * ~c_idx_eval
                        bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval
                        bno_acbothhas_idx = b_missing_idx_eval * a_idx_eval * c_idx_eval

                        bno_aonlyhas = self.autoencoder1.encoder(x1_train[bno_aonlyhas_idx])
                        bno_aonlyhas, _ = self.a2b(bno_aonlyhas)

                        bno_conlyhas = self.autoencoder3.encoder(x3_train[bno_conlyhas_idx])
                        bno_conlyhas, _ = self.c2b(bno_conlyhas)

                        bno_acbothhas_1 = self.autoencoder1.encoder(x1_train[bno_acbothhas_idx])
                        bno_acbothhas_2 = self.autoencoder3.encoder(x3_train[bno_acbothhas_idx])
                        bno_acbothhas = (self.a2b(bno_acbothhas_1)[0] + self.c2b(bno_acbothhas_2)[0]) / 2.0

                        latent_code_b_eval[bno_aonlyhas_idx] = bno_aonlyhas
                        latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas
                        latent_code_b_eval[bno_acbothhas_idx] = bno_acbothhas

                    # for view c
                    if c_missing_idx_eval.sum() != 0:
                        cno_aonlyhas_idx = c_missing_idx_eval * a_idx_eval * ~b_idx_eval
                        cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval
                        cno_abbothhas_idx = c_missing_idx_eval * a_idx_eval * b_idx_eval

                        cno_aonlyhas = self.autoencoder1.encoder(x1_train[cno_aonlyhas_idx])
                        cno_aonlyhas, _ = self.a2c(cno_aonlyhas)

                        cno_bonlyhas = self.autoencoder2.encoder(x2_train[cno_bonlyhas_idx])
                        cno_bonlyhas, _ = self.b2c(cno_bonlyhas)

                        cno_abbothhas_1 = self.autoencoder1.encoder(x1_train[cno_abbothhas_idx])
                        cno_abbothhas_2 = self.autoencoder2.encoder(x2_train[cno_abbothhas_idx])
                        cno_abbothhas = (self.a2c(cno_abbothhas_1)[0] + self.b2c(cno_abbothhas_2)[0]) / 2.0

                        latent_code_c_eval[cno_aonlyhas_idx] = cno_aonlyhas
                        latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas
                        latent_code_c_eval[cno_abbothhas_idx] = cno_abbothhas

                    # fill the existing views
                    latent_code_a_eval[a_idx_eval] = a_latent_eval
                    latent_code_b_eval[b_idx_eval] = b_latent_eval
                    latent_code_c_eval[c_idx_eval] = c_latent_eval

                    # recovered fusion representation
                    latent_fusion = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval],
                                              dim=1).cpu().numpy()

                    scores = clustering.get_score([latent_fusion], Y_list, accumulated_metrics['acc'],
                                                  accumulated_metrics['nmi'],
                                                  accumulated_metrics['ARI'], accumulated_metrics['f-mea'])
                    logger.info("\033[2;29m" + 'trainingset_view_concat ' + str(scores) + "\033[0m")

                    self.autoencoder1.train(), self.autoencoder2.train(), self.autoencoder3.train()
                    self.a2b.train(), self.b2a.train()
                    self.b2c.train(), self.c2b.train()
                    self.a2c.train(), self.c2a.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][
            -1]

    def train_coreview(self, config, logger, accumulated_metrics, x1_train, x2_train, x3_train, Y_list, mask, optimizer,
                       device):
        """Training the model with cove view for clustering

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              x1_train: data of view 1
              x2_train: data of view 2
              x3_train: data of view 3
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari


        """
        epochs_total = config['training']['epoch']
        batch_size = config['training']['batch_size']

        flag = torch.LongTensor([1, 1, 1]).to(device)
        flag = (mask == flag).int()
        flag = ((flag[:, 1] + flag[:, 0] + flag[:, 2]) == 3)
        train_view1 = x1_train[flag]
        train_view2 = x2_train[flag]
        train_view3 = x3_train[flag]

        for k in range(epochs_total):
            X1, X2, X3 = shuffle(train_view1, train_view2, train_view3)
            all0, all1, all2, all_icl, map1, map2 = 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, batch_x3, batch_No in next_batch_3view(X1, X2, X3, batch_size):
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)
                z_half3 = self.autoencoder3.encoder(batch_x3)

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
                recon3 = F.mse_loss(self.autoencoder3.decoder(z_half3), batch_x3)

                reconstruction_loss = recon1 + recon2 + recon3

                # Instance-level Contrastive Loss
                loss_icl1 = instance_contrastive_Loss(z_half1, z_half2, config['training']['alpha'])
                loss_icl2 = instance_contrastive_Loss(z_half1, z_half3, config['training']['alpha'])
                loss_icl = (loss_icl1 + 0.1 * loss_icl2) / 2

                # Cross-view Dual-Prediction Loss
                a2b, _ = self.a2b(z_half1)
                b2a, _ = self.b2a(z_half2)
                a2c, _ = self.a2c(z_half1)
                c2a, _ = self.c2a(z_half3)

                pre1 = F.mse_loss(a2b, z_half2)
                pre2 = F.mse_loss(b2a, z_half1)
                pre3 = F.mse_loss(a2c, z_half3)
                pre4 = F.mse_loss(c2a, z_half1)

                dualprediction_loss = (pre1 + pre2 + pre3 + pre4) / 2

                all_loss = loss_icl + reconstruction_loss * config['training']['lambda2']

                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda1'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                all0 += all_loss.item()
                all1 += recon1.item()
                all2 += recon2.item()
                map1 += pre1.item()
                map2 += pre2.item()
                all_icl += loss_icl.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss = {:.4f}===> Reconstruction loss = {:.4f} " \
                     "===> Map loss = {:.4f} ===> Map loss = {:.4f} ===> loss_icl = {:.4e} ===> all loss = {:.4e}" \
                .format((k + 1), epochs_total, all1, all2, map1, map2, all_icl, all0)
            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval(), self.autoencoder3.eval()
                    self.a2b.eval(), self.b2a.eval()
                    self.a2c.eval(), self.c2a.eval()

                    a_idx_eval = mask[:, 0] == 1
                    b_idx_eval = mask[:, 1] == 1
                    c_idx_eval = mask[:, 2] == 1
                    a_missing_idx_eval = mask[:, 0] == 0
                    b_missing_idx_eval = mask[:, 1] == 0
                    c_missing_idx_eval = mask[:, 2] == 0

                    a_latent_eval = self.autoencoder1.encoder(x1_train[a_idx_eval])
                    b_latent_eval = self.autoencoder2.encoder(x2_train[b_idx_eval])
                    c_latent_eval = self.autoencoder3.encoder(x3_train[c_idx_eval])

                    latent_code_a_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_b_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)
                    latent_code_c_eval = torch.zeros(x3_train.shape[0], config['Autoencoder']['arch3'][-1]).to(
                        device)

                    if a_missing_idx_eval.sum() != 0:
                        ano_bonlyhas_idx = a_missing_idx_eval * b_idx_eval * ~c_idx_eval
                        ano_conlyhas_idx = a_missing_idx_eval * c_idx_eval * ~b_idx_eval
                        ano_bcbothhas_idx = a_missing_idx_eval * b_idx_eval * c_idx_eval

                        ano_bonlyhas = self.autoencoder2.encoder(x2_train[ano_bonlyhas_idx])
                        ano_bonlyhas, _ = self.b2a(ano_bonlyhas)

                        ano_conlyhas = self.autoencoder3.encoder(x3_train[ano_conlyhas_idx])
                        ano_conlyhas, _ = self.c2a(ano_conlyhas)

                        ano_bcbothhas_1 = self.autoencoder2.encoder(x2_train[ano_bcbothhas_idx])
                        ano_bcbothhas_2 = self.autoencoder3.encoder(x3_train[ano_bcbothhas_idx])
                        ano_bcbothhas = (self.b2a(ano_bcbothhas_1)[0] + self.c2a(ano_bcbothhas_2)[0]) / 2.0

                        latent_code_a_eval[ano_bonlyhas_idx] = ano_bonlyhas
                        latent_code_a_eval[ano_conlyhas_idx] = ano_conlyhas
                        latent_code_a_eval[ano_bcbothhas_idx] = ano_bcbothhas

                    if b_missing_idx_eval.sum() != 0:
                        #  to recover view b, utilizing the core view unless core view is missing
                        bno_ahas_idx = b_missing_idx_eval * a_idx_eval

                        bno_conlyhas_idx = b_missing_idx_eval * c_idx_eval * ~a_idx_eval

                        bno_ahas = self.autoencoder1.encoder(x1_train[bno_ahas_idx])
                        bno_ahas, _ = self.a2b(bno_ahas)

                        # predicting twice
                        bno_conlyhas = self.autoencoder3.encoder(x3_train[bno_conlyhas_idx])
                        bno_conlyhas, _ = self.c2a(bno_conlyhas)
                        bno_conlyhas, _ = self.a2b(bno_conlyhas)

                        latent_code_b_eval[bno_ahas_idx] = bno_ahas
                        latent_code_b_eval[bno_conlyhas_idx] = bno_conlyhas

                    if c_missing_idx_eval.sum() != 0:
                        cno_ahas_idx = c_missing_idx_eval * a_idx_eval
                        cno_bonlyhas_idx = c_missing_idx_eval * b_idx_eval * ~a_idx_eval

                        cno_aonlyhas = self.autoencoder1.encoder(x1_train[cno_ahas_idx])
                        cno_aonlyhas, _ = self.a2c(cno_aonlyhas)

                        cno_bonlyhas = self.autoencoder2.encoder(x2_train[cno_bonlyhas_idx])
                        cno_bonlyhas, _ = self.b2a(cno_bonlyhas)
                        cno_bonlyhas, _ = self.a2c(cno_bonlyhas)

                        latent_code_c_eval[cno_ahas_idx] = cno_aonlyhas
                        latent_code_c_eval[cno_bonlyhas_idx] = cno_bonlyhas

                    latent_code_a_eval[a_idx_eval] = a_latent_eval
                    latent_code_b_eval[b_idx_eval] = b_latent_eval
                    latent_code_c_eval[c_idx_eval] = c_latent_eval

                    latent_fusion = torch.cat([latent_code_a_eval, latent_code_b_eval, latent_code_c_eval],
                                              dim=1).cpu().numpy()

                    scores = clustering.get_score([latent_fusion], Y_list, accumulated_metrics['acc'],
                                                  accumulated_metrics['nmi'],
                                                  accumulated_metrics['ARI'], accumulated_metrics['f-mea'])
                    logger.info("\033[2;29m" + 'trainingset_view_concat ' + str(scores) + "\033[0m")

                    self.autoencoder1.train(), self.autoencoder2.train(), self.autoencoder3.train()
                    self.a2b.train(), self.b2a.train()
                    self.a2c.train(), self.c2a.train()

        return accumulated_metrics['acc'][-1], accumulated_metrics['nmi'][-1], accumulated_metrics['ARI'][
            -1]


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