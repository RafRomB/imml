import random
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ._aux_monet import _best_samples_to_add, _which_sample_to_remove, _which_view_to_add_to_module, \
    _which_view_to_remove_from_module, _score_of_split_module, _weight_of_split_and_add_view, \
    _weight_of_split_and_remove_view, _weight_of_new_module, _top_samples_to_switch, \
    _weight_of_spreading_module, _weight_of_merged_modules, _Globals, Sample, _Module, View
from ..utils.utils import check_Xs


class MONET(BaseEstimator, ClassifierMixin):
    r"""
    Use StandardScaler before.

    Parameters
    ----------
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

    Attributes
    ----------
    features_view_mean_list_ : array-like of shape (n_views,)
        The mean value of each feature in the corresponding view, if value='mean'

    Examples
    --------
    >>> from imvc.datasets import load_incomplete_nutrimouse
    >>> from imvc.transformers import FillMissingViews
    >>> Xs = load_incomplete_nutrimouse(p = [0.2, 0.5])
    >>> transformer = FillMissingViews(value = 'mean')
    >>> transformer.fit_transform(Xs)
    """


    def __init__(self, init_modules: dict = None, iters: int = 500, num_of_seeds: int = 10,
                 num_of_samples_in_seed: int = 10, min_mod_size: int = 10, max_sams_per_action: int = 10,
                 percentile_remove_edge: int = None, random_state: int = None):
        self.init_modules = init_modules
        self.iters = iters
        self.num_of_seeds = num_of_seeds
        self.num_of_samples_in_seed = num_of_samples_in_seed
        self.min_mod_size = min_mod_size
        self.max_sams_per_action = max_sams_per_action
        self.percentile_remove_edge = percentile_remove_edge
        self.random_state = random_state

        # a list of the actions considered by MONET in each iteration. Each action correponds to one function in the list.
        self.functions = [_best_samples_to_add, _which_sample_to_remove, _which_view_to_add_to_module,
                          _which_view_to_remove_from_module, _score_of_split_module, _weight_of_split_and_add_view,
                          _weight_of_split_and_remove_view, _weight_of_new_module, _top_samples_to_switch,
                          _weight_of_spreading_module, _weight_of_merged_modules]
        self.functions_names = ["best_samples_to_add", "which_sample_to_remove", "which_view_to_add_to_module",
                                "which_view_to_remove_from_module", "score_of_split_module", "weight_of_split_and_add_view",
                                "weight_of_split_and_remove_view", "weight_of_new_module", "top_samples_to_switch",
                                "weight_of_spreading_module", "weight_of_merged_modules"]



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
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
        glob_var = _Globals(len(self.functions))
        data = {X_idx : np.fill_diagonal(X.corr().values, 0) for X_idx, X in enumerate(Xs)}
        glob_var = self._create_env(glob_var = glob_var, data = data,
                                    percentile_remove_edge = self.percentile_remove_edge)
        glob_var.min_mod_size = self.min_mod_size
        glob_var.max_samps_per_action = self.max_sams_per_action

        if self.init_modules is None:
            glob_var = self._get_seeds(glob_var, num_of_seeds=self.num_of_seeds,
                                       num_of_samples_in_seed=self.num_of_samples_in_seed)
        else:
            glob_var = self._create_seeds_from_solution(glob_var, self.init_modules)

        for _, some_mod in glob_var.modules.copy().items():
            if len(some_mod.samples) < self.min_mod_size:
                glob_var.kill_module(some_mod)

        total_weight = sum(mod.get_weight() for mod in glob_var.modules.values())
        converged_modules = {}
        did_action = False
        mod_index = 0
        iterations = 0

        return self


    def transform(self, Xs):
        r"""
        Transform the input data by filling missing samples.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        transformed_Xs : list of array-likes, shape (n_samples, n_features)
            The transformed data with filled missing samples.
        """

        Xs = check_Xs(Xs, allow_incomplete=True)
        return _


    def _create_env(self, glob_var, data, percentile_remove_edge):
        """
        Create all the variables used during MONET's run:
        modules, view, etc, and associating them with a Global instance.
        """
        all_sam_names = (list(data.values())[i].columns.values for i in range(len(data)))
        for sample in all_sam_names:
            glob_var.samples.update({sample: Sample(sample)})

        for view, dat in data.items():
            self.view = view
            graph, means, covs, percentile = self.build_a_graph_similarity(dat)
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
            glob_var.views.update({view: View(graph=graph, name=view)})
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







