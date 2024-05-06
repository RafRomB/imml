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

load_dataset <- function(dataset_name, return_y, shuffle) {
  
  
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

algorithms = c("IntNMF", "COCA", "JIVE")

results = read.csv(TIME_RESULTS_PATH, row.names = 1)

for (dataset_name in datasets) {
  names <- strsplit(dataset_name, "_")[[1]]
  if ("simulated" %in% names) {
    names <- paste(names, collapse = "_")
  }
  x_name <- names[1]
  y_name <- ifelse(length(names) > 1, names[2], "0")

  data = LoadDataset.load_dataset(dataset_name=x_name, return_y=True, shuffle=True)
  Xs <- data$Xs
  y <- data$y
  y <- y[[y_name]]
  n_clusters <- length(unique(y))
  
  for (alg_name in algorithms) {
    time_execution <- results[alg_name, dataset_name]
    if (time_execution > 0 | is.na(time_execution)) {
      next
    }
    cat(paste(dataset_name, alg_name, Sys.time(), sep = "\t"), file=TIME_LOGS_PATH,append=TRUE)

    tryCatch({
      start_time <- Sys.time()
      if (alg_name == "IntNMF") {
        set.seed(42)
        clusters <- 
      } else if (alg_name == "COCA") {
        set.seed(42)
        clusters <- 
      } else if (alg_name == "JIVE") {
        set.seed(42)
        clusters <- 
      }
      elapsed_time <- difftime(Sys.time(), start_time)
    }, error = function(e) {
      errors_dict <- c(errors_dict, paste(class(e), e))
      cat(paste(dataset_name, alg_name, Sys.time(), sep = "\t"), file=TIME_ERRORS_PATH,append=TRUE))
      elapsed_time <- NA
    })

    results[alg_name, dataset_name] <- elapsed_time
    write.csv(results, file = TIME_RESULTS_PATH)
  }
  
  cat("Completed successfully!\n")
  cat("Completed successfully", file=TIME_LOGS_PATH,append=TRUE)
}
