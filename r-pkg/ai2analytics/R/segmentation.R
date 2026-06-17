## Entity segmentation pipeline. Mirrors ai2analytics.templates.segmentation
## but written natively in R using R6 + base stats + cluster.

#' Build a segmentation configuration.
#'
#' Externalises every schema decision so the same pipeline class works across
#' regions and brands.
#'
#' @param analysis_name        Free-text label for the run.
#' @param col_entity_id        Name of the column holding the entity identifier.
#' @param feature_columns      Character vector of numeric columns to cluster on.
#'                             If empty, the pipeline auto-selects all numeric
#'                             columns except the entity ID.
#' @param exclude_columns      Numeric columns to skip when auto-selecting.
#' @param n_segments           Integer number of clusters when not auto-selecting k.
#' @param method               One of \code{"kmeans"}, \code{"hierarchical"},
#'                             \code{"auto"}.
#' @param normalize            Whether to normalise features before clustering.
#' @param normalization_method One of \code{"standard"} (z-score), \code{"minmax"},
#'                             \code{"robust"} (median/IQR).
#' @param handle_missing       One of \code{"median"}, \code{"mean"}, \code{"zero"},
#'                             \code{"drop"}.
#' @param auto_select_k        If \code{TRUE}, search over \code{k_range}.
#' @param k_range              Integer length-2 vector \code{c(low, high)}.
#' @param output_csv           Optional path to write the assignments CSV.
#'
#' @return A list with class \code{"SegmentationConfig"}.
#'
#' @examples
#' cfg <- SegmentationConfig(
#'   analysis_name   = "demo",
#'   col_entity_id   = "npi",
#'   feature_columns = c("IL_17_TRX_L12M", "IL_23_TRX_L12M"),
#'   n_segments      = 4
#' )
#' @export
SegmentationConfig <- function(analysis_name = "segmentation",
                               col_entity_id = "entity_id",
                               feature_columns = character(),
                               exclude_columns = character(),
                               n_segments = 4L,
                               method = c("kmeans", "hierarchical", "auto"),
                               normalize = TRUE,
                               normalization_method = c("standard", "minmax", "robust"),
                               handle_missing = c("median", "mean", "zero", "drop"),
                               auto_select_k = FALSE,
                               k_range = c(2L, 10L),
                               output_csv = "") {
  method               <- match.arg(method)
  normalization_method <- match.arg(normalization_method)
  handle_missing       <- match.arg(handle_missing)

  cfg <- list(
    analysis_name        = analysis_name,
    col_entity_id        = col_entity_id,
    feature_columns      = as.character(feature_columns),
    exclude_columns      = as.character(exclude_columns),
    n_segments           = as.integer(n_segments),
    method               = method,
    normalize            = isTRUE(normalize),
    normalization_method = normalization_method,
    handle_missing       = handle_missing,
    auto_select_k        = isTRUE(auto_select_k),
    k_range              = as.integer(k_range),
    output_csv           = output_csv
  )
  class(cfg) <- c("SegmentationConfig", "list")
  cfg
}

#' Validate a SegmentationConfig.
#' @param cfg A SegmentationConfig list.
#' @param dataframes Optional named list of data frames; presence of
#'        \code{entity_data} relaxes the table-name requirement.
#' @return Character vector of error messages (empty if valid).
#' @keywords internal
.validate_seg_cfg <- function(cfg, dataframes = NULL) {
  errors <- character()
  has_inmem <- !is.null(dataframes) && "entity_data" %in% names(dataframes)
  if (!has_inmem) {
    errors <- c(errors, "dataframes must contain `entity_data` for the R port.")
  }
  if (cfg$n_segments < 2) {
    errors <- c(errors, "n_segments must be >= 2.")
  }
  if (cfg$auto_select_k) {
    if (cfg$k_range[1] < 2 || cfg$k_range[2] < cfg$k_range[1]) {
      errors <- c(errors, sprintf("k_range invalid: %s", paste(cfg$k_range, collapse = ", ")))
    }
  }
  errors
}


#' Entity segmentation pipeline (R6 class).
#'
#' Stages:
#' \enumerate{
#'   \item Load (in-memory data only in this R port).
#'   \item Prepare features (impute, normalise).
#'   \item Fit clusters (KMeans / hierarchical / auto).
#'   \item Build assignments + per-segment profiles.
#'   \item Write CSV (optional).
#' }
#'
#' @section Methods:
#' \describe{
#'   \item{\code{$run(cfg, dataframes)}}{Run the full pipeline. Returns a list
#'         with \code{assignments}, \code{profiles}, and \code{summary_stats}.}
#'   \item{\code{$dashboard()}}{Open a base-R diagnostic plot of the last run.}
#' }
#'
#' @examples
#' df  <- generate_us_hcp(seed = 42)$hcp_reference
#' cfg <- SegmentationConfig(analysis_name = "demo", col_entity_id = "npi",
#'                           feature_columns = c("IL_17_TRX_L12M", "IL_23_TRX_L12M"),
#'                           n_segments = 4)
#' out <- SegmentationPipeline$new()$run(cfg, dataframes = list(entity_data = df))
#' head(out$assignments)
#' @export
SegmentationPipeline <- R6::R6Class(
  "SegmentationPipeline",
  public = list(
    name        = "segmentation",
    description = "Entity segmentation using KMeans or hierarchical clustering",
    last_output = NULL,
    last_features = NULL,

    initialize = function() {
      invisible(self)
    },

    run = function(cfg, dataframes) {
      stopifnot(inherits(cfg, "SegmentationConfig"))
      errs <- .validate_seg_cfg(cfg, dataframes)
      if (length(errs)) stop(paste("Config validation failed:\n  ",
                                   paste(errs, collapse = "\n  ")))

      message(sprintf("\n=== Pipeline: %s ===", cfg$analysis_name))

      # 1. Load
      df <- as.data.frame(dataframes$entity_data)
      if (!cfg$col_entity_id %in% colnames(df)) {
        stop(sprintf("Entity ID column '%s' not in data. Available: %s",
                     cfg$col_entity_id, paste(colnames(df), collapse = ", ")))
      }

      # Auto-detect numeric features if not provided.
      feats <- cfg$feature_columns
      if (length(feats) == 0) {
        numeric_cols <- vapply(df, is.numeric, logical(1))
        feats <- setdiff(names(df)[numeric_cols],
                         c(cfg$col_entity_id, cfg$exclude_columns))
      }
      missing_feats <- setdiff(feats, colnames(df))
      if (length(missing_feats)) {
        stop(sprintf("Feature columns missing: %s",
                     paste(missing_feats, collapse = ", ")))
      }
      message(sprintf("Loaded %d entities, %d features", nrow(df), length(feats)))

      # 2. Prepare features
      ids <- df[[cfg$col_entity_id]]
      mat_df <- df[, feats, drop = FALSE]
      imp <- .impute(mat_df, cfg$handle_missing)
      mat_df <- imp$df
      ids <- ids[imp$keep]
      mat <- as.matrix(sapply(mat_df, as.numeric))
      mat[is.na(mat)] <- 0

      if (cfg$normalize) {
        mat <- switch(cfg$normalization_method,
                      standard = .standardize(mat),
                      minmax   = .minmax(mat),
                      robust   = .robust(mat))
      }
      private$.last_matrix <- mat
      private$.last_ids    <- ids
      private$.last_feats  <- feats

      # 3. Fit
      fit <- private$.fit(mat, cfg)
      message(sprintf("Method=%s  k=%d  silhouette=%.4f",
                      fit$method, fit$k, fit$silhouette))

      # 4. Build output
      orig_features <- df[, feats, drop = FALSE]
      orig_features[[cfg$col_entity_id]] <- df[[cfg$col_entity_id]]
      orig_features <- orig_features[match(ids, orig_features[[cfg$col_entity_id]]), ,
                                     drop = FALSE]
      orig_features$SEGMENT <- fit$labels

      profiles <- aggregate(
        orig_features[, feats, drop = FALSE],
        by = list(SEGMENT = orig_features$SEGMENT),
        FUN = function(x) mean(as.numeric(x), na.rm = TRUE)
      )
      profiles <- tibble::as_tibble(profiles)

      assignments <- tibble::tibble(!!cfg$col_entity_id := ids,
                                    SEGMENT = as.integer(fit$labels))
      sizes <- as.list(table(fit$labels))
      sizes <- stats::setNames(as.integer(sizes), names(sizes))

      summary_stats <- list(
        n_entities       = nrow(assignments),
        n_segments       = fit$k,
        method           = fit$method,
        silhouette_score = fit$silhouette,
        segment_sizes    = sizes,
        k_scores         = fit$k_scores
      )

      output <- list(
        assignments   = assignments,
        profiles      = profiles,
        summary_stats = summary_stats
      )
      class(output) <- c("SegmentationOutput", "list")
      self$last_output   <- output
      self$last_features <- list(matrix = mat, ids = ids, feature_names = feats)

      # 5. Write CSV
      if (nzchar(cfg$output_csv)) {
        .ensure_parent(cfg$output_csv)
        utils::write.csv(assignments, cfg$output_csv, row.names = FALSE)
        message(sprintf("Wrote CSV: %s", cfg$output_csv))
      }

      invisible(output)
    },

    dashboard = function() {
      out <- self$last_output
      if (is.null(out)) {
        message("Run the pipeline first.")
        return(invisible(NULL))
      }
      mat <- self$last_features$matrix
      labels <- out$assignments$SEGMENT
      op <- graphics::par(mfrow = c(1, 2)); on.exit(graphics::par(op))
      if (ncol(mat) >= 2) {
        # Use first two PC axes if more than 2 features
        coords <- if (ncol(mat) > 2) {
          stats::prcomp(mat, center = FALSE, scale. = FALSE)$x[, 1:2]
        } else mat
        graphics::plot(coords, col = labels + 1, pch = 16, cex = 0.5,
             xlab = "Component 1", ylab = "Component 2",
             main = "Segments (PCA)")
      }
      sizes <- out$summary_stats$segment_sizes
      graphics::barplot(sizes, col = "steelblue", border = "black",
                        main = "Segment sizes", xlab = "Segment", ylab = "Count")
      invisible(NULL)
    }
  ),
  private = list(
    .last_matrix = NULL,
    .last_ids    = NULL,
    .last_feats  = NULL,

    .fit = function(mat, cfg) {
      methods <- if (cfg$method == "auto") c("kmeans", "hierarchical") else cfg$method

      best <- NULL
      for (m in methods) {
        if (cfg$auto_select_k) {
          ks <- seq(cfg$k_range[1], cfg$k_range[2])
          k_scores <- numeric(length(ks))
          for (i in seq_along(ks)) {
            labs <- private$.fit_one(mat, ks[i], m)
            k_scores[i] <- .silhouette_score(mat, labs)
          }
          names(k_scores) <- as.character(ks)
          best_k <- ks[which.max(k_scores)]
        } else {
          best_k <- cfg$n_segments
          k_scores <- NULL
        }
        labels <- private$.fit_one(mat, best_k, m)
        sil <- .silhouette_score(mat, labels)
        cand <- list(method = m, k = best_k, labels = labels,
                     silhouette = sil, k_scores = k_scores)
        if (is.null(best) || cand$silhouette > best$silhouette) best <- cand
      }
      best
    },

    .fit_one = function(mat, k, method) {
      if (method == "kmeans") {
        as.integer(stats::kmeans(mat, centers = k, nstart = 10, iter.max = 50)$cluster)
      } else {
        d <- stats::dist(mat)
        h <- stats::hclust(d, method = "ward.D2")
        as.integer(stats::cutree(h, k = k))
      }
    }
  )
)
