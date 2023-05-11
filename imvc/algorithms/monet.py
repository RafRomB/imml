import copy
import operator
import random
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from mixem.distribution import MultivariateNormalDistribution
from mixem import mix_expectation_maximization

from ._aux_monet import _best_samples_to_add, _which_sample_to_remove, _which_view_to_add_to_module, \
    _which_view_to_remove_from_module, _score_of_split_module, _weight_of_split_and_add_view, \
    _weight_of_split_and_remove_view, _weight_of_new_module, _top_samples_to_switch, \
    _weight_of_spreading_module, _weight_of_merged_modules, _Globals, _Sample, _Module, _View, _switch_2_samples
from utils import check_Xs


class MONET(BaseEstimator, ClassifierMixin):
    r"""
    Use StandardScaler before.

    The output is a set of modules, where each module is a subset of the samples. Modules are disjoint, and not all
    samples necessarily belong to a module. Samples not belonging to a module are called lonely. Each module M is
    characterized by its samples, denoted samples(M), and by a set of omics that it covers, denoted omics(M).
    Intuitively, samples(M) are similar to one another in omics(M).

    MONET works in two phases. It first constructs an edge-weighted graph per omic, such that nodes are samples
    and weights correspond to the similarity between samples in that omic. In the second phase, it detects modules
    by looking for heavy subgraphs common to multiple omic graphs.

    Parameters
    ----------
    num_repeats : int (default=15)
        Times the algorithm will be repeated in order to avoid suboptimal (local maximum) solutions. The best solution
        will be returned.
    init_modules : dict (default=None)
        an optional module initialization for MONET. A dict mapping between module names to sample ids. All modules
        are initialized to cover all views. Set to None to use MONET's seed finding algorithm for initialization.
    iters : int (default=500)
        Maximal number of iterations.
    num_of_seeds : int (default=10)
        Number of seeds to create in MONET's module initialization algorithm.
    num_of_samples_in_seed : int (default=10)
        Number of samples to put in each seeds to create in MONET's module initialization algorithm.
    min_mod_size : int (default=10)
        Minimal size (number of samples) for a MONET module.
    max_samples_per_action : int (default=10)
        Maximal number of samples in a single MONET action (maximal number of samples added to a module or replaced
        between modules in a single action).
    percentile_remove_edge : int (default=None)
        Only edges with weight percentile above (for positive weights) or below (for negative weights) this percentile
        are kept in the graph. For example, percentile_remove_edge=90 keeps only the 10% edges with highest positive
        weight and lowest negative weight in the graph. one keeps all edges in the graph.
    random_state : int (default=None)
        Determines random number generation for centroid initialization. Use an int to make the randomness
        deterministic.
    verbose : bool, default=False
        Verbosity mode.
    n_jobs : int (default=None)
        The number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means
        using all processors.
.
    Attributes
    ----------
    labels_ : array-like of shape (n_views,)
        The mean value of each feature in the corresponding view, if value='mean'
    glob_var_ : dict
        Module names to Module objects mapping. Every module instance includes its set of
        samples (under the "samples" attribute) and its set of views (the "views" attribute).
    total_weight_ : float
        Sum of the weights (similarity between samples within the module) of all modules.
    view_graphs_ : list of dataframes
        Graph of each view.

    Examples
    --------
    >>> from imvc.datasets import load_incomplete_nutrimouse
    >>> from imvc.algorithms import MONET
    >>> Xs = load_incomplete_nutrimouse(p = [0.2, 0.5])
    >>> estimator = MONET()
    >>> estimator.fit_predict(Xs)
    """


    def __init__(self, num_repeats: int = 15, init_modules: dict = None, iters: int = 500, num_of_seeds: int = 10,
                 num_of_samples_in_seed: int = 10, min_mod_size: int = 10, max_sams_per_action: int = 10,
                 percentile_remove_edge: int = None, random_state: int = None, verbose: bool = False, n_jobs: int = None):
        self.num_repeats = num_repeats
        self.init_modules = init_modules
        self.iters = iters
        self.num_of_seeds = num_of_seeds
        self.num_of_samples_in_seed = num_of_samples_in_seed
        self.min_mod_size = min_mod_size
        self.max_sams_per_action = max_sams_per_action
        self.percentile_remove_edge = percentile_remove_edge
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        # a list of the actions considered by MONET in each iteration. Each action correponds to one function in the list.
        self.functions = [_best_samples_to_add, _which_sample_to_remove, _which_view_to_add_to_module,
                          _which_view_to_remove_from_module, _score_of_split_module, _weight_of_split_and_add_view,
                          _weight_of_split_and_remove_view, _weight_of_new_module, _top_samples_to_switch,
                          _weight_of_spreading_module, _weight_of_merged_modules]


    def fit(self, Xs, y=None):
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
        Xs = check_Xs(Xs, allow_incomplete=True)
        data = self._process_data(Xs= Xs)
        solutions = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_run)(data = data, init_modules = self.init_modules, iters = self.iters,
                                      num_of_seeds = self.num_of_seeds,
                                      num_of_samples_in_seed = self.num_of_samples_in_seed,
                                      min_mod_size = self.min_mod_size, max_sams_per_action = self.max_sams_per_action,
                                      percentile_remove_edge = self.percentile_remove_edge,
                                      random_state = self.random_state + n_time if self.random_state is not None else self.random_state,
                                      verbose = self.verbose) for n_time in range(self.num_repeats)
        )
        solutions = {idx:i for idx,i in enumerate(solutions)}
        best_sol = {key: value['total_weight'] for key,value in solutions.items()}
        best_sol = max(best_sol.items(), key=operator.itemgetter(1))[0]
        best_sol = solutions[best_sol]
        glob_var, total_weight = best_sol['glob_var'], best_sol['total_weight']
        labels, view_graphs = self._post_processing(glob_var = glob_var)
        self.labels_ = labels
        self.glob_var_ = glob_var
        self.total_weight_ = total_weight
        self.view_graphs_ = view_graphs
        return self


    def predict(self, Xs):
        r"""
        Return clustering results for new samples.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        labels : list of array-likes, shape (n_samples,)
            The predicted data.
        """
        Xs = check_Xs(Xs, allow_incomplete=True)
        data = self._process_data(Xs= Xs)
        global_var = copy.deepcopy(self.glob_var_)
        labels = []
        for sam in samples:
            all_sam_names = set()
            for i in range(len(data.values())):
                all_sam_names.update(list(data.values())[i].columns.values)
            for sample in all_sam_names:
                glob_var.samples.update({sample: _Sample(sample)})

            sample = _Sample(sam)
            mod_w_dict = {}
            for mod_idx,mod in global_var.get_modules().items():
                start_weight = mod.get_weight()
                tmp_weight = mod.add_sample(sample)
                current_weight = tmp_weight - start_weight
                mod.remove_sample(sample)
                mod_w_dict[mod_idx] = current_weight
            if all([i < 0 for i in mod_w_dict.values()]):
                label = None
            else:
                label = max(mod_w_dict, key=mod_w_dict.get)
            labels.append(label)
        labels = np.array(labels)
        return labels


    def fit_predict(self, Xs):
        r"""
        Fit the model and return clustering results.
        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        labels : list of array-likes, shape (n_samples,)
            The predicted data.
        """

        labels = self.fit(Xs).labels_
        return labels


    def _single_run(self, data, init_modules, iters, num_of_seeds, num_of_samples_in_seed, min_mod_size,
                    max_sams_per_action, percentile_remove_edge, random_state, verbose):
        r"""

        """
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
        glob_var = _Globals(len(self.functions))
        glob_var = self._create_env(glob_var = glob_var, data = data,
                                    percentile_remove_edge = percentile_remove_edge)
        glob_var.min_mod_size = min_mod_size
        glob_var.max_samps_per_action = max_sams_per_action

        if init_modules is None:
            glob_var = self._get_seeds(glob_var, num_of_seeds=num_of_seeds,
                                       num_of_samples_in_seed=num_of_samples_in_seed)
        else:
            glob_var = self._create_seeds_from_solution(glob_var, init_modules)

        for _, some_mod in glob_var.modules.copy().items():
            if len(some_mod.samples) < min_mod_size:
                glob_var.kill_module(some_mod)

        total_weight = sum(mod.get_weight() for mod in glob_var.modules.values())
        converged_modules = {}
        did_action = False
        iterations = 0

        while iterations < iters:
            prev_weight = total_weight

            active_module_names = list(sorted(set(glob_var.modules.keys()) - set(converged_modules.keys())))
            if len(active_module_names) == 0:
                if not did_action:
                    if verbose:
                        print("converged, total score: {}.".format(total_weight))
                    break
                else:
                    converged_modules = {}
                    did_action = False
                    active_module_names = list(sorted(glob_var.modules.keys()))
            mod_name = random.choice(active_module_names)
            mod = glob_var.modules[mod_name]

            max_res = self._get_next_step(mod, glob_var)
            glob_var = self._exec_next_step(mod, max_res, glob_var)
            for _, some_mod in glob_var.modules.copy().items():
                if len(some_mod.get_samples()) <= 1 or not some_mod.get_views():
                    glob_var.kill_module(some_mod)
                    if verbose:
                        print('removing zombie module')

            total_weight = sum([mod.get_weight() for name, mod in glob_var.modules.items()])
            iterations += 1

            if (iterations % 10 == 0) and verbose:
                print("iteration: " + str(iterations))
                print("num of modules: " + str(len(glob_var.modules)))
                print("total_weight: " + str(total_weight))
                print("actions: " + str(glob_var.actions))

            # Assert module sizes
            for _, some_mod in glob_var.modules.copy().items():
                assert len(some_mod.samples) >= min_mod_size

            if total_weight <= prev_weight or max_res[1][0] == -float("inf"):
                if mod_name in glob_var.modules:
                    converged_modules.update({mod_name: glob_var.modules[mod_name]})
            else:  # the score deviates from the score we expected
                if not (abs(total_weight - prev_weight - max_res[1][0]) < 0.01):
                    # This signifies a bug and should never occur:
                    # that the difference in the objective function from the
                    # previous iteration is different from the difference
                    # the algorithm expected for the function.
                    import pdb;
                    pdb.set_trace()
                did_action = True
                assert abs(total_weight - prev_weight - max_res[1][0]) < 0.01
                did_action = True
        for mod_name, mod in glob_var.modules.copy().items():
            if mod.get_size() <= glob_var.min_mod_size and not self._is_mod_significant(mod, glob_var):
                if verbose:
                    print("module {} with samples {} on views {} is not significant.".format((mod_name, mod),
                                                                                              mod.get_samples(),
                                                                                              mod.get_views().keys()))
                glob_var.kill_module(mod)

        return {"glob_var": glob_var, "total_weight": total_weight}


    @staticmethod
    def _process_data(Xs: list):
        """gets raw data and return a list of similarity matrices"""
        data = {}
        for X_idx, X in enumerate(Xs):
            X_t = X.copy().T
            X_t.columns = X_t.columns.astype(str)
            X_t = X_t.corr()
            np.fill_diagonal(X_t.values, 0)
            data[str(X_idx)] = X_t
        return data


    def _create_env(self, glob_var, data, percentile_remove_edge):
        """
        Create all the variables used during MONET's run:
        modules, view, etc, and associating them with a Global instance.
        """
        all_sam_names = set()
        for i in range(len(data.values())):
            all_sam_names.update(list(data.values())[i].columns.values)
        for sample in all_sam_names:
            glob_var.samples.update({sample: _Sample(sample)})

        for view, dat in data.items():
            self.view = view
            graph, means, covs, percentile = self._build_a_graph_similarity(dat)
            if percentile_remove_edge is not None:
                all_weights = []
                for edge in graph.edges:
                    all_weights.append(graph.edges[edge]['weight'])
                all_weights_array = np.array(all_weights)
                positive_thresh = np.percentile(all_weights_array[all_weights_array > 0], percentile_remove_edge)
                negative_thresh = np.percentile(all_weights_array[all_weights_array < 0], 100 - percentile_remove_edge)
                all_edges = []
                for edge in graph.edges:
                    all_edges.append(edge)
                for edge in all_edges:
                    cur_weight = graph.edges[edge]['weight']
                    if (cur_weight > 0 and cur_weight < positive_thresh) or (
                            cur_weight < 0 and cur_weight > negative_thresh):
                        graph.remove_edge(edge[0], edge[1])

            cur_graph_sams = set(graph.nodes)
            missing_sams = all_sam_names - cur_graph_sams
            for missing_sam in missing_sams:
                graph.add_node(missing_sam)
            glob_var.views.update({view: _View(graph=graph, name=view)})
            glob_var.gmm_params.update({view: {'mean': means, 'cov': covs, 'percentile': percentile}})
        return glob_var


    def _create_seeds_from_solution(self, glob_var, init_modules):
        for mod_name, sam_ids in init_modules.items():
            views = glob_var.views
            sam_dict = {}
            for sam_id in sam_ids:
                sam_dict[sam_id] = glob_var.samples[sam_id]
            mod_weight = 0
            for view in views.values():
                mod_weight += view.graph.subgraph(list(sam_dict.keys())).size('weight')
            _Module(glob_var=glob_var, samples=sam_dict, views=views, weight=mod_weight)
        return glob_var


    def _get_seeds(self, glob_var, num_of_seeds=3, num_of_samples_in_seed=10):
        """
        Create seed modules.
        """
        lst = list(glob_var.views.items())
        lst.sort(key=lambda x: x[0])
        views_list = [view for name, view in lst]
        sam_list = list(glob_var.samples.keys())
        adj = np.zeros((len(sam_list), len(sam_list)))
        for name, view in lst:
            adj += nx.adjacency_matrix(view.graph.subgraph(sam_list), nodelist=sam_list)
        adj = pd.DataFrame(adj, index=sam_list, columns=sam_list)
        joined_subgraph = nx.from_pandas_adjacency(adj)
        view_graphs = [joined_subgraph]

        for i in range(num_of_seeds):

            view_graph = view_graphs[0]
            cur_nodes = list(sorted(view_graph.nodes()))
            adj = list(view_graph.adjacency())

            if len(cur_nodes) == 0:
                break

            rand_sam_index = random.randint(0, len(cur_nodes) - 1)
            rand_sam_name = cur_nodes[rand_sam_index]
            rand_sam_in_adj = [sam[0] for sam in adj].index(rand_sam_name)
            neighbors = [(key, adj[rand_sam_in_adj][1][key]['weight']) for key in adj[rand_sam_in_adj][1]]
            neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:(num_of_samples_in_seed - 1)]
            nodes = {rand_sam_name: glob_var.samples[rand_sam_name]}
            for nei in neighbors:
                if nei[1] > 0 and nei[0] != rand_sam_name:
                    nodes.update({nei[0]:glob_var.samples[nei[0]]})
            mod_weight = view_graph.subgraph(list(nodes.keys())).size('weight')
            if mod_weight > 0 and len(nodes) > 1 and len(nodes) >= glob_var.min_mod_size:
                _Module(glob_var=glob_var, samples=nodes, views=[view for view in views_list], weight=mod_weight)
                for k in range(len(view_graphs)):
                    view_graph = view_graphs[k]
                    remaining_nodes = list(sorted(set(cur_nodes) - set(nodes.keys())))
                    view_graphs[k] = view_graph.subgraph(remaining_nodes)
        return glob_var


    def _build_a_graph_similarity(self, distances):
        g = nx.from_numpy_array(distances.values)
        mapping = {i: j for i,j in enumerate(distances.columns)}
        nx.relabel_nodes(g, mapping, False)
        return g, [], [], 0


    def _get_next_step(self, mod, glob_var):
        """
        this function decided what is the next action that will be executed.
        """
        max_res = (-1, (-float("inf"), None))
        for func_i in range(len(self.functions)):
            if func_i <= 9:  # only one module needed
                tmp = self.functions[func_i](mod, glob_var)
                if tmp[0] > max_res[1][0]:
                    max_res = (func_i, tmp)
            else:
                for mod2 in glob_var.modules.values():
                    if mod2 == mod:
                        continue
                    tmp = self.functions[func_i](mod, mod2, glob_var)
                    if tmp[0] > max_res[1][0]:
                        max_res = (func_i, tmp)
        return max_res


    def _exec_next_step(self, mod, max_res, glob_var):
        """
        this function actually performs an action, given that the
        algorithm already decided what the next action will be.
        """
        if max_res[1][0] == -float("inf") or max_res[1][0] < 0:
            return glob_var
        func_i = max_res[0]
        glob_var.actions[func_i] += 1
        if func_i == 0:  # add
            for sample in max_res[1][1]:
                mod.add_sample(sample)
        elif func_i == 1:  # remove
            mod.remove_sample(max_res[1][1])
            if len(mod.get_samples()) <= 1:
                glob_var = glob_var.kill_module(mod)
        elif func_i == 2:  # add view
            mod.add_view(max_res[1][1], glob_var)
        elif func_i == 3:  # remove view
            mod.remove_view(max_res[1][1], glob_var)
        elif func_i == 4:  # split
            glob_var = mod.split_module(max_res[1][1][1], glob_var)
        elif func_i == 5:  # split and add view
            glob_var = mod.split_and_add_view(view=max_res[1][1][0], sub_nodes=max_res[1][1][1], glob_var=glob_var)
        elif func_i == 6:  # split and remove view
            glob_var = mod.split_and_remove_view(view=max_res[1][1][0], sub_nodes=max_res[1][1][1], glob_var=glob_var)
        elif func_i == 7:  # create new module
            new_mod = _Module(glob_var)
            new_mod.add_view(max_res[1][1][1], glob_var)
            for sam in max_res[1][1][0]:
                new_mod.add_sample(glob_var.samples[sam])
        elif func_i == 8:  # transfer
            sams = [(sam, mod2) for sam, weight, mod2 in max_res[1][1]]
            for sam, mod2 in sams:
                _switch_2_samples(glob_var.samples[sam], mod, mod2, glob_var)
        elif func_i == 9:  # spread module
            mod.spread_module(max_res[1][1], glob_var)
        elif func_i == 10:  # merge
            glob_var = mod.merge_with_module(max_res[1], glob_var)
        return glob_var


    def _is_mod_significant(mod, glob_var, percentile=95, iterations=500):
        """
        Assess the statisitcal significance of a module by sampling modules or similar size.
        """
        draws = [0 for i in range(iterations)]
        mod_size = len(mod.get_samples())
        if mod_size <= 1:
            return False
        for i in range(iterations):
            samps = random.sample(glob_var.samples.keys(), mod_size)
            lst = list(mod.get_views().items())
            lst.sort(key=lambda x: x[0])
            for name, view in lst:
                draws[i] += view.graph.subgraph(samps).size('weight')
        num_to_beat = np.percentile(draws, percentile)
        return mod.get_weight() > num_to_beat


    @staticmethod
    def _post_processing(glob_var):
        labels = [[sample, mod_id] for mod_id, module in glob_var.modules.items() for sample in module.samples]
        labels = pd.DataFrame(labels)
        labels = labels.set_index(0)
        sams_without_mods = pd.DataFrame(None, index= glob_var.samples.keys())
        labels = pd.concat([labels, sams_without_mods.loc[sams_without_mods.index.difference(labels.index)]])
        view_graphs = [pd.DataFrame(nx.to_numpy_array(view.graph)) for view in glob_var.views.values()]
        return labels, view_graphs


    def _get_em_graph_per_omic(self, Xs: list, input_em_rets = None):
        """gets raw data and return a list of similarity matrices"""
        sim_data = self._process_data(Xs=Xs)
        sim_data = list(sim_data.values())
        for i, cur_sim in enumerate(sim_data):
            if input_em_rets is None:
                n_clusters = list(range(2, 11))
                scores = [self._compute_clustering_scores(data = cur_sim, k = k) for k in n_clusters]
                num_clusters = n_clusters[np.argmax(scores)]
            else:
                #todo
            num_to_sample = len(cur_sim)
            chosen = np.random.choice(num_to_sample, size=num_to_sample, replace=False)
            chosen_sims_mat = cur_sim.loc[chosen, chosen]
            chosen_sims = chosen_sims_mat[np.triu_indices(num_to_sample, k=1)]
            if input_em_rets is not None:
                #todo
                em_ret = input_em_rets[i][0]
            else:
                all_em_rets = []
                for _ in range(20):
                    mixture_model = mix_expectation_maximization(
                        data=chosen_sims, distributions=[MultivariateNormalDistribution] * 2,
                        k=2, max_iterations=1e5, random_seed=42
                    )
                    all_em_rets.append(mixture_model)
                best_model_idx = np.argmax([model.log_likelihood() for model in all_em_rets])
                em_ret = all_em_rets[best_model_idx]
        return em_ret, num_clusters


    @staticmethod
    def _compute_clustering_scores(data, k):
        model = SpectralClustering(n_clusters=k, affinity='precomputed')
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        return score







