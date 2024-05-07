RANDOM_STATE = 42
TIME_LIMIT = 100000

RESULTS_FOLDER = 'results'
SUBRESULTS_FOLDER = 'subresults'
SUBRESULTS_PATH = file.path(RESULTS_FOLDER, SUBRESULTS_FOLDER)

COMPLETE_RESULTS_FILE = 'complete_algorithms_evaluation.csv'
INCOMPLETE_RESULTS_FILE = 'incomplete_algorithms_evaluation.csv'
TIME_RESULTS_FILE = 'time_evaluation.csv'
PROFILES_FILE = 'profiles.json'
COMPLETE_RESULTS_PATH = file.path(RESULTS_FOLDER, COMPLETE_RESULTS_FILE)
INCOMPLETE_RESULTS_PATH = file.path(RESULTS_FOLDER, INCOMPLETE_RESULTS_FILE)
TIME_RESULTS_PATH = file.path(RESULTS_FOLDER, TIME_RESULTS_FILE)
PROFILES_PATH = file.path(RESULTS_FOLDER, PROFILES_FILE)

COMPLETE_LOGS_FILE = 'complete_logs.txt'
INCOMPLETE_LOGS_FILE = 'incomplete_logs.txt'
TIME_LOGS_FILE = 'time_logs.txt'
COMPLETE_LOGS_PATH = file.path(RESULTS_FOLDER, COMPLETE_LOGS_FILE)
INCOMPLETE_LOGS_PATH = file.path(RESULTS_FOLDER, INCOMPLETE_LOGS_FILE)
TIME_LOGS_PATH = file.path(RESULTS_FOLDER, TIME_LOGS_FILE)

COMPLETE_ERRORS_FILE = 'complete_errors.txt'
INCOMPLETE_ERRORS_FILE = 'incomplete_errors.txt'
TIME_ERRORS_FILE = 'time_errors.txt'
COMPLETE_ERRORS_PATH = file.path(RESULTS_FOLDER, COMPLETE_ERRORS_FILE)
INCOMPLETE_ERRORS_PATH = file.path(RESULTS_FOLDER, INCOMPLETE_ERRORS_FILE)
TIME_ERRORS_PATH = file.path(RESULTS_FOLDER, TIME_ERRORS_FILE)

load_dataset <- function(dataset_name, return_y, shuffle, seed = FALSE) {
  data_path <- file.path("imvc", "datasets", "data", dataset_name)
  data_files <- list.files(data_path)
  data_files <- data_files[order(data_files)]
  files_to_read <- c()
  for (file in data_files) {
    files_to_read <- c(files_to_read, file.path(data_path, file))
  }
  data_files <- files_to_read[grep(dataset_name, files_to_read) & !grepl("y.csv", files_to_read) & grepl(".csv", files_to_read)]
  Xs <- lapply(data_files, read.csv)
  view_data <- Xs[[1]]
  samples <- row.names(view_data)
  if (shuffle) {
    if (isFALSE(seed)) {
      set.seed(seed)
    }
    samples <- row.names(view_data[sample(1:nrow(view_data)),])
    Xs <- lapply(Xs, function(X) X[samples, ])
  }
  output <- list(X = Xs)
  if (return_y) {
    y <- read.csv(file.path(data_path, paste0(dataset_name, "_y.csv")))
    y <- y[samples, , drop = FALSE]
    output[["y"]] <- y
  }
  return(output)
}

normalize <- function(x, ...) {
  return((x - min(x, ...)) /(max(x, ...) - min(x, ...)))
}


nemo.affinity.graph <- function(raw.data, k=NA) {
  if (is.na(k)) {
    k = as.numeric(lapply(1:length(raw.data), function(i) round(ncol(raw.data[[i]]) / 6)))
  } else if (length(k) == 1) {
    k = rep(k, length(raw.data))
  }
  sim.data = lapply(1:length(raw.data), function(i) {SNFtool::affinityMatrix(SNFtool::dist2(as.matrix(t(raw.data[[i]])),
                                                                          as.matrix(t(raw.data[[i]]))), k[i], 0.5)})
  affinity.per.omic = lapply(1:length(raw.data), function(i) {
    sim.datum = sim.data[[i]]
    non.sym.knn = apply(sim.datum, 1, function(sim.row) {
      returned.row = sim.row
      threshold = sort(sim.row, decreasing = T)[k[i]]
      returned.row[sim.row < threshold] = 0
      row.sum = sum(returned.row)
      returned.row[sim.row >= threshold] = returned.row[sim.row >= threshold] / row.sum
      return(returned.row)
    })
    sym.knn = non.sym.knn + t(non.sym.knn)
    return(sym.knn)
  })
  patient.names = Reduce(union, lapply(raw.data, colnames))
  num.patients = length(patient.names)
  returned.affinity.matrix = matrix(0, ncol = num.patients, nrow=num.patients)
  rownames(returned.affinity.matrix) = patient.names
  colnames(returned.affinity.matrix) = patient.names
  
  shared.omic.count = matrix(0, ncol = num.patients, nrow=num.patients)
  rownames(shared.omic.count) = patient.names
  colnames(shared.omic.count) = patient.names
  
  for (j in 1:length(raw.data)) {
    curr.omic.patients = colnames(raw.data[[j]])
    returned.affinity.matrix[curr.omic.patients, curr.omic.patients] = returned.affinity.matrix[curr.omic.patients, curr.omic.patients] + affinity.per.omic[[j]][curr.omic.patients, curr.omic.patients]
    shared.omic.count[curr.omic.patients, curr.omic.patients] = shared.omic.count[curr.omic.patients, curr.omic.patients] + 1
  }
  
  final.ret = returned.affinity.matrix / shared.omic.count
  lower.tri.ret = final.ret[lower.tri(final.ret)]
  final.ret[shared.omic.count == 0] = mean(lower.tri.ret[!is.na(lower.tri.ret)])
  
  return(final.ret)
}

nemo.clustering <- function(omics.list, num.clusters=NULL, num.neighbors=NA) {
  if (is.null(num.clusters)) {
    num.clusters = NA
  }
  
  graph = nemo.affinity.graph(omics.list, k = num.neighbors)
  if (is.na(num.clusters)) {
    num.clusters = nemo.num.clusters(graph)
  }
  clustering = SNFtool::spectralClustering(graph, num.clusters)
  names(clustering) = colnames(graph)
  return(clustering)
}

datasets = c(
  "simulated_gm",
  "simulated_InterSIM",
  "simulated_netMUG",
  "nutrimouse_genotype",
  "nutrimouse_diet",
  "bbcsport",
  "buaa",
  "metabric",
  "digits",
  "bdgp",
  "tcga",
  "caltech101",
  "nuswide"
)

algorithms = c("IntNMF", "COCA", "jNMF", "NEMO")

results = read.csv(TIME_RESULTS_PATH, row.names = 1)


for (dataset_name in datasets) {
  names <- strsplit(dataset_name, "_")[[1]]
  if ("simulated" %in% names) {
    names <- paste(names, collapse = "_")
  }
  x_name <- names[1]
  y_name <- ifelse(length(names) > 1, names[2], "X0")
  
  
  data = load_dataset(dataset_name=x_name, return_y=T, shuffle=RANDOM_STATE)
  Xs <- data[["X"]]
  y <- data[["y"]]
  y <- y[,y_name]
  n_clusters <- length(unique(y))
  
  for (alg_name in algorithms) {
    if (!alg_name %in% row.names(results)) {
      results[alg_name,] <- 0
    }
    time_execution <- results[alg_name, dataset_name]
    if (time_execution > 0 | is.na(time_execution)) {
      next
    }
    cat(paste("\n", dataset_name, alg_name, Sys.time(), sep = "\t"),
        file=TIME_LOGS_PATH,append=TRUE)

    elapsed_time <- tryCatch({
      start_time <- Sys.time()
      if (alg_name == "IntNMF") {
        train_Xs = lapply(Xs, normalize)
        train_Xs = lapply(train_Xs, as.matrix)
        set.seed(RANDOM_STATE)
        clusters <- IntNMF::nmf.mnnals(dat=train_Xs, k=n_clusters, seed=RANDOM_STATE)$clusters
      } else if (alg_name == "COCA") {
        train_Xs = lapply(Xs, scale)
        set.seed(RANDOM_STATE)
        clusters <- coca::buildMOC(train_Xs, M = length(train_Xs), K = n_clusters)$moc
        clusters <- coca::coca(clusters, K = n_clusters)$clusterLabels
      } else if (alg_name == "jNMF") {
        train_Xs = lapply(Xs, normalize)
        train_Xs = lapply(train_Xs, as.matrix)
        set.seed(RANDOM_STATE)
        clusters <- nnTensor::jNMF(X= train_Xs, J = n_clusters)$W
        set.seed(RANDOM_STATE)
        clusters <- kmeans(clusters, n_clusters)$cluster
      } else if (alg_name == "NEMO") {
        train_Xs = lapply(Xs, t)
        train_Xs = lapply(train_Xs, scale)
        set.seed(RANDOM_STATE)
        clusters <- nemo.clustering(train_Xs, num.clusters= n_clusters)
      }
      stopifnot(length(y) == length(clusters))
      elapsed_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    }, error = function(e) {
      cat(paste("\n", dataset_name, alg_name, paste(class(e), e), Sys.time(), sep = "\t"),
          file=TIME_ERRORS_PATH,append=TRUE)
      return(NA)
    })

    results[alg_name, dataset_name] <- elapsed_time
    write.csv(results, file = TIME_RESULTS_PATH)
  }
}
cat("Completed successfully!")
cat("Completed successfully", file=TIME_LOGS_PATH,append=TRUE)
