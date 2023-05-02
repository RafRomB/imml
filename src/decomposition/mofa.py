from mofapy2.run.entry_point import entry_point
import os
import contextlib
from mofapy2.core.BayesNet import BayesNet, StochasticBayesNet
from mofapy2.core.utils import nans
import sys
import pandas as pd
from time import time
import numpy as np


class MOFA():
    
    def __init__(self, factors : int = 10, random_state : int = None, verbose = False, data_options = {}, data_matrix = {}, model_options = {}, train_options = {}, stochastic_options = {},
                 covariates = {}, smooth_options = {}):
        self.factors = factors
        self.random_state = random_state
        self.verbose = verbose        
        if self.verbose:
            self.estimator = entry_point()
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                self.estimator = entry_point()
        self.data_options_args = data_options
        self.data_matrix_args = data_matrix
        self.model_options_args = model_options
        self.train_options_args = train_options
        self.stochastic_options_args = stochastic_options
        self.covariates_args = covariates
        self.smooth_options_args = smooth_options
        self.transform_ = None

        
    def fit(self, X, y = None):
        if self.verbose:
            self.run_mofa(data = [[view] for view in X])
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                self.run_mofa(data = [[view] for view in X])
        return self

    def transform(self, X):
        if self.transform_ == "pandas":
            projected_data = [pd.DataFrame(np.dot(view, self.estimator.model.nodes['W'].getExpectations()[idx]['E']), index = view.index) for idx, view in enumerate(X)]
            projected_data = pd.concat(projected_data, axis = 1)
        else:
            projected_data = [np.dot(view, self.estimator.model.nodes['W'].getExpectations()[idx]['E']) for idx, view in enumerate(X)]
            projected_data = np.concatenate(projected_data, axis = 1)
        return projected_data
    
    
    def run_mofa(self, data):
        self.estimator.set_data_options(**self.data_options_args)
        self.estimator.set_data_matrix(data = data, **self.data_matrix_args)
        self.estimator.set_model_options(factors = self.factors, **self.model_options_args)
        self.estimator.set_train_options(seed = self.random_state, verbose = self.verbose, **self.train_options_args)
        self.estimator.set_stochastic_options(**self.stochastic_options_args)
        if self.covariates_args:
            self.estimator.set_covariates(**self.covariates_args)
            self.estimator.set_smooth_options(**self.smooth_options_args)
        self.estimator.build()
        if isinstance(self.estimator.model, BayesNet):
            self.estimator.model = ModifiedBayesNet(self.estimator.model.dim, self.estimator.model.nodes)
        elif isinstance(self.estimator.model, StochasticBayesNet):
            self.estimator.model = ModifiedStochasticBayesNet(self.estimator.model.dim, self.estimator.model.nodes)
        self.estimator.run()
        return None
    
    
    def set_output(self, *, transform=None):
        self.transform_ = "pandas"
        return self

        
class ModifiedBayesNet(BayesNet):
    
    
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
                            "- ELBO decomposition:  "
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
                
                
class ModifiedStochasticBayesNet(StochasticBayesNet):
    
    
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
                        "- ELBO decomposition:  "
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
        