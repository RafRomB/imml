from mofapy2.run.entry_point import entry_point
import os
import contextlib
from mofapy2.core.BayesNet import BayesNet, StochasticBayesNet
from mofapy2.core.utils import nans
import sys
import pandas as pd
from time import time
import numpy as np


class MOFA:
    r"""
    MOFA is a factor analysis model that provides a general framework for the integration of (originally, multi-omic
    data sets) incomplete multi-view datasets, in an unsupervised fashion. Intuitively, MOFA can be viewed as a
    versatile and statistically rigorous generalization of principal component analysis to multi-views data. Given
    several data matrices with measurements of multiple -views data types on the same or on overlapping sets of
    samples, MOFA infers an interpretable low-dimensional representation in terms of a few latent factors.

    Parameters
    ----------
    factors : int, default=10
        The number of clusters to generate. If it is not provided, it will use the default one from the algorithm.
    data_options : dict (default={})
        Data processing options, such as scale_views and scale_groups.
    data_matrix : dict (default={})
        Keys such as likelihoods, view_names, etc.
    model_options : dict (default={})
        Model options, such as ard_factors or ard_weights.
    train_options : dict (default={})
        Keys such as iter, tolerance.
    stochastic_options : dict (default={})
        Stochastic variational inference options, such as learning rate or batch size.
    covariates : dict (default={})
        Slot to store sample covariate for training in MEFISTO. Keys are sample_cov and covariates_names.
    smooth_options : dict (default={})
        options for smooth inference, such as scale_cov or model_groups.
    random_state : int (default=None)
        Determines random number generation for centroid initialization. Use an int to make the randomness
        deterministic.
    verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    mofa_ : mofa object
        Entry point as the original library. This can be used for data analysis and explainability.

    References
    ----------
    [paper1] Argelaguet R, Velten B, Arnol D, Dietrich S, Zenz T, Marioni JC, Buettner F, Huber W, Stegle O
    (2018). “Multi‐Omics Factor Analysis—a framework for unsupervised integration of multi‐omics data sets.” Molecular
    Systems Biology, 14. doi:10.15252/msb.20178124.
    [paper2] Argelaguet R, Arnol D, Bredikhin D, Deloro Y, Velten B, Marioni JC, Stegle O (2020). “MOFA+: a statistical
    framework for comprehensive integration of multi-modal single-cell data.” Genome Biology, 21.
    doi:10.1186/s13059-020-02015-1.
    [url] https://biofam.github.io/MOFA2/index.html

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.algorithms import MOFA
    >>> Xs = LoadDataset.load_incomplete_nutrimouse(p = 0.2)
    >>> pipeline = MOFA().fit(Xs)
    >>> transformed_Xs = pipeline.transform(Xs)
    """

    
    def __init__(self, factors : int = 10, data_options = {}, data_matrix = {}, model_options = {}, train_options = {},
                 stochastic_options = {}, covariates = {}, smooth_options = {}, random_state : int = None,
                 verbose = False):
        self.factors = factors
        self.random_state = random_state
        self.verbose = verbose        
        if self.verbose:
            self.mofa_ = entry_point()
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                self.mofa_ = entry_point()
        self.data_options_args = data_options
        self.data_matrix_args = data_matrix
        self.model_options_args = model_options
        self.train_options_args = train_options
        self.stochastic_options_args = stochastic_options
        self.covariates_args = covariates
        self.smooth_options_args = smooth_options
        self.transform_ = None

        
    def fit(self, Xs, y = None):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : array-like, shape (n_samples,)
            Labels for each sample. Only used by supervised algorithms.

        Returns
        -------
        self :  returns and instance of self.
        """
        if self.verbose:
            self._run_mofa(data = [[view] for view in Xs])
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                self._run_mofa(data = [[view] for view in Xs])
        return self

    def transform(self, Xs):
        r"""
        Project data into the learned space.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        transformed_Xs : list of array-likes, shape (n_samples, n_features)
            The projected data.
        """
        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(np.dot(view, self.mofa_.model.nodes['W'].getExpectations()[idx]['E']), index = view.index) for idx, view in enumerate(Xs)]
            transformed_Xs = pd.concat(transformed_Xs, axis = 1)
        else:
            transformed_Xs = [np.dot(view, self.mofa_.model.nodes['W'].getExpectations()[idx]['E']) for idx, view in enumerate(Xs)]
            transformed_Xs = np.concatenate(transformed_Xs, axis = 1)
        return transformed_Xs
    
    
    def _run_mofa(self, data):
        self.mofa_.set_data_options(**self.data_options_args)
        self.mofa_.set_data_matrix(data = data, **self.data_matrix_args)
        self.mofa_.set_model_options(factors = self.factors, **self.model_options_args)
        self.mofa_.set_train_options(seed = self.random_state, verbose = self.verbose, **self.train_options_args)
        self.mofa_.set_stochastic_options(**self.stochastic_options_args)
        if self.covariates_args:
            self.mofa_.set_covariates(**self.covariates_args)
            self.mofa_.set_smooth_options(**self.smooth_options_args)
        self.mofa_.build()
        if isinstance(self.mofa_.model, BayesNet):
            self.mofa_.model = _ModifiedBayesNet(self.mofa_.model.dim, self.mofa_.model.nodes)
        elif isinstance(self.mofa_.model, StochasticBayesNet):
            self.mofa_.model = _ModifiedStochasticBayesNet(self.mofa_.model.dim, self.mofa_.model.nodes)
        self.mofa_.run()
        return None
    
    
    def set_output(self, *, transform=None):
        self.transform_ = "pandas"
        return self

        
class _ModifiedBayesNet(BayesNet):
    
    
    def iterate(self):
        """Method to start iterating and updating the variables using the VB algorithm"""

        # Define some variables to monitor training
        nodes = list(self.getVariationalNodes().keys())
        elbo = pd.DataFrame(
            data=nans((self.options["maxiter"] + 1, len(nodes) + 1)),
            columns=nodes + ["total"],
        )
        number_factors = nans((self.options["maxiter"] + 1))
        iter_time = nans((self.options["maxiter"] + 1))
        # keep track of factor-wise training statistics (attribute as needs to be accounted for in factor dropping)
        # if 'Sigma' in self.nodes.keys():
        #     self.lscales = pd.DataFrame(data = nans((self.options['maxiter'], self.dim['K'])), columns = range(self.dim['K']))
        #     self.scales = pd.DataFrame(data = nans((self.options['maxiter'], self.dim['K'])), columns = range(self.dim['K']))

        # Precompute
        converged = False
        convergence_token = 1
        elbo.iloc[0] = self.precompute()
        number_factors[0] = self.dim["K"]
        iter_time[0] = 0.0

        try:
            for i in range(1, self.options["maxiter"]):
                t = time()

                # Remove inactive factors
                if (i >= self.options["start_drop"]) and (
                    i % self.options["freq_drop"]
                ) == 0:
                    if self.options["drop"]["min_r2"] is not None:
                        self.removeInactiveFactors(**self.options["drop"])
                    number_factors[i] = self.dim["K"]

                # Update node by node, with E and M step merged
                t_updates = time()
                for node in self.options["schedule"]:
                    if (node == "ThetaW" or node == "ThetaZ") and i < self.options[
                        "start_sparsity"
                    ]:
                        continue
                    self.nodes[node].update()
                t_updates = time() - t_updates

                # Calculate Evidence Lower Bound
                if (i >= self.options["start_elbo"]) and (
                    (i - self.options["start_elbo"]) % self.options["freqELBO"] == 0
                ):
                    t_elbo = time()
                    elbo.iloc[i] = self.calculateELBO()
                    t_elbo = time() - t_elbo

                    # Check convergence using the ELBO
                    if i == self.options["start_elbo"]:
                        delta_elbo = elbo.iloc[i]["total"] - elbo.iloc[0]["total"]
                    else:
                        delta_elbo = (
                            elbo.iloc[i]["total"]
                            - elbo.iloc[i - self.options["freqELBO"]]["total"]
                        )

                    # Print ELBO monitoring
                    if not self.options["quiet"]:
                        print(
                            "Iteration %d: time=%.2f, ELBO=%.2f, deltaELBO=%.3f (%.8f%%), Factors=%d"
                            % (
                                i,
                                time() - t,
                                elbo.iloc[i]["total"],
                                delta_elbo,
                                100 * abs(delta_elbo / elbo.iloc[0]["total"]),
                                (self.dim["K"]),
                            )
                        )
                        if delta_elbo < 0 and not self.options["stochastic"]:
                            print("Warning, lower bound is decreasing...\a")

                    # Print ELBO decomposed by node and variance explained
                    if self.options["verbose"]:
                        print(
                            "- ELBO modules:  "
                            + "".join(
                                [
                                    "%s=%.2f  " % (k, v)
                                    for k, v in elbo.iloc[i].drop("total").items()
                                ]
                            )
                        )
                        print(
                            "- Time spent in ELBO computation: %.1f%%"
                            % (100 * t_elbo / (t_updates + t_elbo))
                        )

                    # Assess convergence
                    if (
                        i > self.options["start_elbo"]
                        and i > self.options["min_iter"]
                        and not self.options["forceiter"]
                    ):
                        convergence_token, converged = self.assess_convergence(
                            delta_elbo, elbo.iloc[0]["total"], convergence_token
                        )
                        if converged:
                            number_factors = number_factors[:i]
                            elbo = elbo[:i]
                            iter_time = iter_time[:i]
                            print("\nConverged!\n")
                            break

                # Do not calculate lower bound
                else:
                    if not self.options["quiet"]:
                        print(
                            "Iteration %d: time=%.2f, Factors=%d"
                            % (i, time() - t, self.dim["K"])
                        )

                # Print other statistics
                if self.options["verbose"]:
                    self.print_verbose_message(i)

                iter_time[i] = time() - t

                # Flush (we need this to print when running on the cluster)
                sys.stdout.flush()

            self.trained = True

        except KeyboardInterrupt:
            self.trained = False

        finally:
            # Finish by collecting the training statistics
            self.train_stats = {
                "time": iter_time,
                "number_factors": number_factors,
                "elbo": elbo["total"].values,
                "elbo_terms": elbo.drop(labels="total", axis= 1),
            }
            if "Sigma" in self.nodes.keys():
                tmp = self.nodes["Sigma"].getParameters()  # save only last iteration
                self.train_stats["length_scales"] = tmp["l"]
                self.train_stats["scales"] = tmp["scale"]
                self.train_stats["Kg"] = tmp["Kg"]

                # self.train_stats['length_scales'] = self.lscales
                # self.train_stats['scales'] = self.scales
                
                
    def precompute(self):
        # Precompute terms
        for n in self.nodes:
            self.nodes[n].precompute(self.options)

        # Precompute ELBO
        for node in self.nodes["Y"].getNodes():
            node.TauTrick = False  # important to do this for ELBO computation
        elbo = self.calculateELBO()
        for node in self.nodes["Y"].getNodes():
            node.TauTrick = True

        if self.options["verbose"]:
            print("ELBO before training:")
            print(
                "".join(
                    ["%s=%.2f  " % (k, v) for k, v in elbo.drop("total").items()]
                )
                + "\nTotal: %.2f\n" % elbo["total"]
            )
        else:
            if not self.options["quiet"]:
                print("ELBO before training: %.2f \n" % elbo["total"])

        return elbo
                
                
class _ModifiedStochasticBayesNet(StochasticBayesNet):
    
    
    def iterate(self):
        """Method to start iterating and updating the variables using the VB algorithm"""

        # Define some variables to monitor training
        nodes = list(self.getVariationalNodes().keys())
        elbo = pd.DataFrame(
            data=nans((self.options["maxiter"] + 1, len(nodes) + 1)),
            columns=nodes + ["total"],
        )
        number_factors = nans((self.options["maxiter"] + 1))
        iter_time = nans((self.options["maxiter"] + 1))
        # if 'Sigma' in self.nodes.keys():
        #     self.lscales = pd.DataFrame(data = nans((self.options['maxiter'], self.dim['K'])), columns = range(self.dim['K']))
        #     self.scales = pd.DataFrame(data = nans((self.options['maxiter'], self.dim['K'])), columns = range(self.dim['K']))

        # Precompute
        converged = False
        convergence_token = 1
        elbo.iloc[0] = self.precompute()
        number_factors[0] = self.dim["K"]
        iter_time[0] = 0.0
        iter_count = 0

        # Print stochastic settings before training
        print("Using stochastic variational inference with the following parameters:")
        print(
            "- Batch size (fraction of samples): %.2f\n- Forgetting rate: %.2f\n- Learning rate: %.2f\n- Starts at iteration: %d \n"
            % (
                100 * self.options["batch_size"],
                self.options["forgetting_rate"],
                self.options["learning_rate"],
                self.options["start_stochastic"],
            )
        )
        ix = None

        for i in range(1, self.options["maxiter"]):
            t = time()

            # Sample mini-batch and define step size for stochastic inference
            if i >= self.options["start_stochastic"]:
                ix, epoch = self.sample_mini_batch_no_replace(
                    i - (self.options["start_stochastic"] - 1)
                )
                ro = self.step_size2(epoch)
            else:
                ro = 1.0

            # Doesn't really make a big difference...
            # if i==self.options["start_stochastic"]:
            #     self.options['schedule'].pop( self.options['schedule'].index("Z") )
            #     self.options['schedule'].insert(1,"Z")

            # Remove inactive factors
            if (i >= self.options["start_drop"]) and (
                i % self.options["freq_drop"]
            ) == 0:
                if self.options["drop"]["min_r2"] is not None:
                    self.removeInactiveFactors(**self.options["drop"])
                number_factors[i] = self.dim["K"]

            # Update node by node, with E and M step merged
            t_updates = time()
            for node in self.options["schedule"]:
                if (node == "ThetaW" or node == "ThetaZ") and i < self.options[
                    "start_sparsity"
                ]:
                    continue
                self.nodes[node].update(ix, ro)
            t_updates = time() - t_updates

            # # Save lengthscales from Sigma node
            # if 'Sigma' in self.nodes.keys():
            #     tmp = self.nodes['Sigma'].getParameters()
            #     self.lscales.iloc[i] = tmp['l']
            #     self.scales.iloc[i] = tmp['scale']
            #     self.Kg.iloc[i] = tmp['Kg']

            # Calculate Evidence Lower Bound
            if (i >= self.options["start_elbo"]) and (
                (i - self.options["start_elbo"]) % self.options["freqELBO"] == 0
            ):
                t_elbo = time()
                elbo.iloc[i] = self.calculateELBO()
                t_elbo = time() - t_elbo

                # Check convergence using the ELBO
                if i == self.options["start_elbo"]:
                    delta_elbo = elbo.iloc[i]["total"] - elbo.iloc[0]["total"]
                else:
                    delta_elbo = (
                        elbo.iloc[i]["total"]
                        - elbo.iloc[i - self.options["freqELBO"]]["total"]
                    )

                # Print ELBO monitoring
                print(
                    "Iteration %d: time=%.2f, ELBO=%.2f, deltaELBO=%.3f (%.9f%%), Factors=%d"
                    % (
                        i,
                        time() - t,
                        elbo.iloc[i]["total"],
                        delta_elbo,
                        100 * abs(delta_elbo / elbo.iloc[0]["total"]),
                        (self.dim["K"]),
                    )
                )
                if delta_elbo < 0 and not self.options["stochastic"]:
                    print("Warning, lower bound is decreasing...\a")

                # Print ELBO decomposed by node and variance explained
                if self.options["verbose"]:
                    print(
                        "- ELBO modules:  "
                        + "".join(
                            [
                                "%s=%.2f  " % (k, v)
                                for k, v in elbo.iloc[i].drop("total").items()
                            ]
                        )
                    )
                    print(
                        "- Time spent in ELBO computation: %.1f%%"
                        % (100 * t_elbo / (t_updates + t_elbo))
                    )

                # Assess convergence
                if i > self.options["start_elbo"] and not self.options["forceiter"]:
                    convergence_token, converged = self.assess_convergence(
                        delta_elbo, elbo.iloc[0]["total"], convergence_token
                    )
                    if converged:
                        number_factors = number_factors[:i]
                        elbo = elbo[:i]
                        iter_time = iter_time[:i]
                        print("\nConverged!\n")
                        break

            # Do not calculate lower bound
            else:
                print(
                    "Iteration %d: time=%.2f, Factors=%d"
                    % (i, time() - t, self.dim["K"])
                )

            # Print other statistics
            if i >= (self.options["start_stochastic"]):
                print("- Step size: %.3f" % ro)

            if self.options["verbose"]:
                self.print_verbose_message(i)
            # print("")

            iter_time[i] = time() - t
            iter_count += 1

            # Flush (we need this to print when running on the cluster)
            sys.stdout.flush()

        if iter_count + 1 == self.options["maxiter"]:
            print(
                "\nMaximum number of iterations reached: {}\n".format(
                    self.options["maxiter"]
                )
            )

        # Finish by collecting the training statistics
        self.train_stats = {
            "time": iter_time,
            "number_factors": number_factors,
            "elbo": elbo["total"].values,
            "elbo_terms": elbo.drop(labels="total", axis= 1),
        }
        if "Sigma" in self.nodes.keys():
            tmp = self.nodes["Sigma"].getParameters()  # save only last iteration
            self.train_stats["length_scales"] = tmp["l"]
            self.train_stats["scales"] = tmp["scale"]
            self.train_stats["Kg"] = tmp["Kg"]

            # self.train_stats['length_scales'] = self.lscales
            # self.train_stats['scales'] = self.scales

        self.trained = True
        