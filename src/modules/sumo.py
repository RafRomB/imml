from utils import DatasetUtils
from sumo.modes.prepare.similarity import feature_to_adjacency
from sumo.network import MultiplexNet
from sumo.modes.run.solvers.unsupervised_sumo import UnsupervisedSumoNMF
from sumo.modes.run.run import run_thread_wrapper
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from sumo.utils import extract_ncut
import numpy as np
import multiprocessing as mp


class SUMO():

    def __init__(self, k: int, method=['euclidean'], missing: list = [0.1], neighbours: float = 0.1, alpha: float = 0.5,
                 sparsity: list = [0.1], repetitions: int = 60,
                 cluster_method: str = "max_value", max_iter: int = 500, tol: float = 1e-5, subsample: float = 0.05,
                 calc_cost: int = 20,
                 h_init: int = None, n_jobs: int = 1, rep: int = 5, random_state: int = None, verbose: bool = False):
        self.method = method
        self.missing = missing
        self.neighbours = neighbours
        self.alpha = alpha
        self.k = k if isinstance(k, list) else [k]
        self.sparsity = sparsity
        self.repetitions = repetitions
        self.cluster_method = cluster_method
        self.max_iter = max_iter
        self.tol = tol
        self.subsample = subsample
        self.calc_cost = calc_cost
        self.h_init = h_init
        self.n_jobs = n_jobs
        self.rep = rep
        self.random_state = random_state
        self.verbose = verbose

        self.graph_ = None
        self.nmf_ = None
        self.priors_ = None

        if self.repetitions < 1:
            raise ValueError("Incorrect value of 'repetitions' parameter")
        if self.n_jobs < 1:
            raise ValueError("Incorrect number of threads")
        if self.subsample > 0.5 or self.subsample < 0:
            # do not allow for removal of more then 50% of samples in each run
            raise ValueError("Incorrect value of 'subsample' parameter")
        if self.rep < 1:
            # number of times additional consensus matrix will be created
            raise ValueError("Incorrect value of 'rep' parameter")
        if self.random_state is not None and self.random_state < 0:
            raise ValueError("Seed value cannot be negative")
        self.runs_per_con = max(round(self.repetitions * 0.8), 1)  # number of runs per consensus matrix creation

        if len(self.k) > 2 or (len(self.k) == 2 and self.k[0] > self.k[1]):
            raise ValueError("Incorrect range of k values")
        elif len(self.k) == 2:
            self.k = list(range(self.k[0], self.k[1] + 1))

    def fit(self, X, y=None):
        if len(self.missing) == 1:
            if self.verbose:
                print(f"#Setting all 'missing' parameters to {self.missing[0]}")
            self.missing = [self.missing[0]] * len(X)
        if len(self.method) == 1:
            self.method = [self.method[0]] * len(X)
        elif len(X) != len(self.method):
            raise ValueError(
                "Number of matrices extracted from input files and number of similarity methods does not correspond")

        all_samples = DatasetUtils.get_sample_views(X=X).index.values
        if self.verbose:
            print(f"Total number of unique samples: {len(all_samples)}")
        self.similarity_ = {}
        adj_matrices = []
        # create adjacency matrices
        for X_idx, X in enumerate(X):
            if self.verbose:
                print(f"#Layer: {X_idx}")
                print(f"Feature matrix: ({X.shape[0]} samples x {X.shape[1]} features)")
            # create adjacency matrix
            a = feature_to_adjacency(X.values, missing=self.missing[X_idx], method=self.method[view_idx],
                                     n=self.neighbours, alpha=self.alpha)
            if self.verbose:
                print(f"Adjacency matrix: ({a.shape} created [similarity method: {self.method[view_idx]}")
            # add matrices to output arrays
            adj_matrices.append(a)
            self.similarity_[str(X_idx)] = a
            self.similarity_[f"f{X_idx}"] = X

        ##################################################################
        if self.h_init is not None:
            if self.h_init >= len(adj_matrices) or self.h_init < 0:
                raise ValueError("Incorrect value of h_init")

        # create multilayer graph
        self.graph = MultiplexNet(adj_matrices=adj_matrices, node_labels=all_samples)
        n_sub_samples = round(all_samples.size * self.subsample)
        if self.verbose:
            print(f"#Number of samples randomly removed in each run: {n_sub_samples} out of {all_samples.size}")
        # create solver
        self.nmf = UnsupervisedSumoNMF(graph=self.graph, nbins=self.repetitions,
                                       bin_size=self.graph.nodes - n_sub_samples, rseed=self.random_state)
        global _sumo_run
        _sumo_run = self  # this solves multiprocessing issue with pickling
        # run factorization for every (eta, k)
        cophenet_list = []
        pac_list = []
        cluster_list = []
        for k in self.k:
            if self.verbose:
                print(f"#K:{k}")
            if self.n_jobs == 1:
                results = [SUMO._run_factorization(sparsity=sparsity, k=k, sumo_run=_sumo_run, verbose=self.verbose) for
                           sparsity in self.sparsity]
                sparsity_order = self.sparsity
            else:
                if self.verbose:
                    print(f"{self.sparsity} processes to run")
                pool = mp.Pool(self.n_jobs)
                results = []
                sparsity_order = []
                iproc = 1
                for res in pool.imap_unordered(run_thread_wrapper, zip(self.sparsity, [k] * len(self.sparsity))):
                    if self.verbose:
                        print(f"- process {iproc} finished")
                    results.append(res[0])
                    sparsity_order.append(res[1])
                    iproc += 1

            # select best result
            best_result = sorted(results, reverse=True)[0]
            best_eta = None

            quality_output = []
            for (result, sparsity) in zip(results, sparsity_order):
                if self.verbose:
                    print(f"#Clustering quality (eta={sparsity}): {result[0]}")
                quality_output.append(np.array([sparsity, result[0]]))
                if result[1] == best_result[1]:
                    best_eta = sparsity

            # summarize results
            assert best_eta is not None
            out_arrays = best_result[1]
            cophenet_list.append(out_arrays["cophenet"])
            pac_list.append(out_arrays["pac"])
            cluster_list.append(out_arrays["clusters"])
        labels = np.stack([i[:, 1] for i in cluster_list], axis=1)
        if len(self.k) == 1:
            cophenet_list, pac_list, cluster_list, labels = cophenet_list[0], pac_list[0], cluster_list[0], labels[:, 0]
        self.cophenet_list_ = cophenet_list
        self.pac_list_ = pac_list
        self.cluster_list_ = cluster_list
        self.labels_ = labels
        return self

    @staticmethod
    def _run_factorization(sparsity: float, k: int, sumo_run, verbose: bool):
        """ Run factorization for set sparsity and number of clusters
        Args:
            sparsity (float): value of sparsity penalty
            k (int): number of clusters
            sumo_run: SumoRun object
        Returns:
            quality (float): assessed quality of cluster structure
            outfile (str): path to .npz output file with results of factorization
        """
        if sumo_run.random_state is not None:
            np.random.seed(sumo_run.random_state)

        # run factorization N times
        results = []
        for repeat in range(sumo_run.repetitions):
            if verbose:
                print(f"#Runing NMF algorithm with sparsity {sparsity} (N={repeat + 1})")
            opt_args = {
                "sparsity_penalty": sparsity,
                "k": k,
                "max_iter": sumo_run.max_iter,
                "tol": sumo_run.tol,
                "calc_cost": sumo_run.calc_cost,
                "bin_id": repeat,
                "h_init": sumo_run.h_init
            }
            result = sumo_run.nmf.factorize(**opt_args)
            # extract computed clusters
            if verbose:
                print(f"#Using {sumo_run.cluster_method} for cluster labels extraction)")
            result.extract_clusters(method=sumo_run.cluster_method)
            results.append(result)

        # consensus graph
        assert len(results) > 0

        all_REs = []  # residual errors
        for run_idx in range(sumo_run.repetitions):
            all_REs.append(results[run_idx].RE)

        out_arrays = {'pac': np.array([]), 'cophenet': np.array([])}
        minRE, maxRE = min(all_REs), max(all_REs)

        for rep in range(sumo_run.rep):
            run_indices = list(np.random.choice(range(len(results)), sumo_run.runs_per_con, replace=False))

            consensus = np.zeros((sumo_run.graph.nodes, sumo_run.graph.nodes))
            weights = np.empty((sumo_run.graph.nodes, sumo_run.graph.nodes))
            weights[:] = np.nan

            all_equal = np.allclose(minRE, maxRE)

            for run_idx in run_indices:
                weight = np.empty((sumo_run.graph.nodes, sumo_run.graph.nodes))
                weight[:] = np.nan
                sample_ids = results[run_idx].sample_ids
                if all_equal:
                    weight[sample_ids, sample_ids[:, None]] = 1.
                else:
                    weight[sample_ids, sample_ids[:, None]] = (maxRE - results[run_idx].RE) / (maxRE - minRE)

                weights = np.nansum(np.stack((weights, weight)), axis=0)
                consensus_run = np.nanprod(np.stack((results[run_idx].connectivity, weight)), axis=0)
                consensus = np.nansum(np.stack((consensus, consensus_run)), axis=0)

            if verbose:
                print(f"#Creating consensus graphs [{rep + 1} out of {sumo_run.rep}]")
            assert not np.any(np.isnan(consensus))
            consensus = consensus / weights

            org_con = consensus.copy()
            consensus[consensus < 0.5] = 0

            # calculate cophenetic correlation coefficient
            dist = pdist(org_con, metric="correlation")
            if np.any(np.isnan(dist)):
                ccc = np.nan
                if verbose:
                    print(
                        "Cannot calculate cophenetic correlation coefficient! Please inspect values in your consensus matrix")
            else:
                ccc = cophenet(linkage(dist, method="complete", metric="correlation"), dist)[0]

            # calculate proportion of ambiguous clustering
            den = (sumo_run.graph.nodes ** 2) - sumo_run.graph.nodes
            num = org_con[(org_con > 0.1) & (org_con < 0.9)].size
            pac = num * (1. / den)

            out_arrays.update({'pac': np.append(out_arrays['pac'], pac),
                               'cophenet': np.append(out_arrays['cophenet'], ccc)})

        if verbose:
            print("#Extracting final clustering result, using normalized cut")
        consensus_labels = extract_ncut(consensus, k=k)

        cluster_array = np.empty((sumo_run.graph.sample_names.shape[0], 2), dtype=np.object)
        # TODO add column with confidence value when investigating soft clustering
        cluster_array[:, 0] = sumo_run.graph.sample_names
        cluster_array[:, 1] = consensus_labels

        clusters_dict = {num: sumo_run.graph.sample_names[list(np.where(consensus_labels == num)[0])] for num in
                         np.unique(consensus_labels)}
        for cluster_idx in sorted(clusters_dict.keys()):
            if verbose:
                print(
                    f"Cluster {cluster_idx} ({len(clusters_dict[cluster_idx])} samples): \n{clusters_dict[cluster_idx]}")

        # calculate quality of clustering for given sparsity
        quality = sumo_run.graph.get_clustering_quality(labels=cluster_array[:, 1])
        # create output file
        conf_array = np.empty((9, 2), dtype=object)
        conf_array[:, 0] = ['method', 'n', 'max_iter', 'tol', 'subsample', 'calc_cost', 'h_init', 'seed', 'sparsity']
        conf_array[:, 1] = [sumo_run.cluster_method, sumo_run.repetitions, sumo_run.max_iter, sumo_run.tol,
                            sumo_run.subsample,
                            sumo_run.calc_cost, np.nan if sumo_run.h_init is None else sumo_run.h_init,
                            np.nan if sumo_run.random_state is None else sumo_run.random_state, sparsity]
        out_arrays.update({
            "clusters": cluster_array,
            "consensus": consensus,
            "unfiltered_consensus": org_con,
            "quality": np.array(quality),
            "samples": sumo_run.graph.sample_names,
            "config": conf_array
        })

        steps_reached = [results[i].steps for i in range(len(results))]
        maxiter_proc = round((sum([step == sumo_run.max_iter for step in steps_reached]) / len(steps_reached)) * 100, 3)
        if verbose:
            print(f"#Reached maximum number of iterations in {maxiter_proc}% of runs")
        if maxiter_proc >= 90:
            if verbose:
                print(f"Consider increasing -max_iter and decreasing -tol to achieve better accuracy")
        out_arrays['steps'] = np.array([steps_reached])
        return quality, out_arrays
