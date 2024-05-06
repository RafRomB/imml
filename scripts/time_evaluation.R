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
  data_files <- files_to_read[grep(dataset_name, files_to_read) & !grepl("y.csv", files_to_read)]
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

algorithms = c("IntNMF", "COCA", "jNMF")

results = read.csv(TIME_RESULTS_PATH, row.names = 1)


for (dataset_name in datasets) {
  names <- strsplit(dataset_name, "_")[[1]]
  if ("simulated" %in% names) {
    names <- paste(names, collapse = "_")
  }
  x_name <- names[1]
  y_name <- ifelse(length(names) > 1, names[2], "X0")
  
  
  data = load_dataset(dataset_name=x_name, return_y=T, shuffle=42)
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
    cat(paste(dataset_name, alg_name, Sys.time(), "\n", sep = "\t"),
        file=TIME_LOGS_PATH,append=TRUE)

    tryCatch({
      start_time <- Sys.time()
      if (alg_name == "IntNMF") {
        set.seed(42)
        clusters <- nmf.mnnals(dat=Xs, k=n_clusters, seed=42)$clusters
      } else if (alg_name == "COCA") {
        set.seed(42)
        clusters <- coca::buildMOC(Xs, M = length(Xs), K = n_clusters)$moc
        clusters <- coca::coca(clusters, K = n_clusters)$clusterLabels
      } else if (alg_name == "jNMF") {
        set.seed(42)
        clusters <- nnTensor::jNMF(X= Xs, J = n_clusters)
        set.seed(42)
        clusters <- kmeans(clusters)
      }
      elapsed_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    }, error = function(e) {
      errors_dict <- c(errors_dict, paste(class(e), e))
      cat(paste(dataset_name, alg_name, Sys.time(), "\n", sep = "\t"),
          file=TIME_ERRORS_PATH,append=TRUE)
      elapsed_time <- NA
    })

    results[alg_name, dataset_name] <- elapsed_time
    write.csv(results, file = TIME_RESULTS_PATH)
  }
  
  cat("Completed successfully!")
  cat("Completed successfully", file=TIME_LOGS_PATH,append=TRUE)
}
