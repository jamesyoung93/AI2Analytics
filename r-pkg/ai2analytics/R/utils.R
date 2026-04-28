## Internal utilities used across the package. Not exported.

#' Standardise (z-score) a numeric matrix column-wise.
#' @keywords internal
.standardize <- function(mat) {
  centers <- apply(mat, 2, mean, na.rm = TRUE)
  scales  <- apply(mat, 2, stats::sd, na.rm = TRUE)
  scales[scales == 0 | is.na(scales)] <- 1
  scaled <- sweep(mat, 2, centers, "-")
  scaled <- sweep(scaled, 2, scales, "/")
  attr(scaled, "centers") <- centers
  attr(scaled, "scales")  <- scales
  scaled
}

#' Min-max scale a numeric matrix column-wise to [0, 1].
#' @keywords internal
.minmax <- function(mat) {
  mins <- apply(mat, 2, min, na.rm = TRUE)
  maxs <- apply(mat, 2, max, na.rm = TRUE)
  rng <- maxs - mins
  rng[rng == 0] <- 1
  sweep(sweep(mat, 2, mins, "-"), 2, rng, "/")
}

#' Robust scale (median / IQR).
#' @keywords internal
.robust <- function(mat) {
  centers <- apply(mat, 2, stats::median, na.rm = TRUE)
  iqrs    <- apply(mat, 2, stats::IQR,    na.rm = TRUE)
  iqrs[iqrs == 0 | is.na(iqrs)] <- 1
  sweep(sweep(mat, 2, centers, "-"), 2, iqrs, "/")
}

#' Impute missing values per column.
#' @keywords internal
.impute <- function(df, method = c("median", "mean", "zero", "drop")) {
  method <- match.arg(method)
  if (method == "drop") {
    keep <- stats::complete.cases(df)
    return(list(df = df[keep, , drop = FALSE], keep = keep))
  }
  for (col in colnames(df)) {
    x <- df[[col]]
    if (any(is.na(x))) {
      fill <- switch(method,
                     median = stats::median(x, na.rm = TRUE),
                     mean   = mean(x, na.rm = TRUE),
                     zero   = 0)
      if (is.na(fill)) fill <- 0
      x[is.na(x)] <- fill
      df[[col]] <- x
    }
  }
  list(df = df, keep = rep(TRUE, nrow(df)))
}

#' Silhouette score wrapper that handles edge cases gracefully.
#' @keywords internal
.silhouette_score <- function(mat, labels) {
  if (length(unique(labels)) < 2) return(0.0)
  d <- stats::dist(mat)
  s <- cluster::silhouette(labels, d)
  if (is.matrix(s)) mean(s[, 3]) else 0.0
}

#' Convert a list of records to a tibble, robust to NULLs.
#' @keywords internal
.records_to_tibble <- function(records) {
  if (length(records) == 0) return(tibble::tibble())
  tibble::as_tibble(do.call(rbind.data.frame, c(records, stringsAsFactors = FALSE)))
}

#' Make a directory for a path, no-op if path has no dirname.
#' @keywords internal
.ensure_parent <- function(path) {
  d <- dirname(path)
  if (nzchar(d) && d != "." && !dir.exists(d)) {
    dir.create(d, recursive = TRUE, showWarnings = FALSE)
  }
  invisible(NULL)
}
