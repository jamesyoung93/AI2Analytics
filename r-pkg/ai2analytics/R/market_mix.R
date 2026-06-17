## Market mix model pipeline. Mirrors ai2analytics.templates.market_mix.

#' Build a Market Mix Model configuration.
#'
#' @param analysis_name      Free-text label for the run.
#' @param col_date           Date column name.
#' @param col_response       Response column (e.g. revenue, sales).
#' @param media_columns      Character vector of media spend columns.
#' @param control_columns    Character vector of control variable columns.
#' @param frequency          One of "weekly", "daily", "monthly".
#' @param default_decay_rate Geometric adstock decay (0 <= rate < 1).
#' @param default_saturation One of "log", "hill", "none".
#' @param hill_half_max      Half-max for Hill saturation.
#' @param hill_steepness     Steepness for Hill saturation.
#' @param include_trend      Add a normalized linear-trend feature.
#' @param include_seasonality Add Fourier seasonality (period = seasonality_period).
#' @param seasonality_period Integer period for the seasonality terms.
#' @param model_type         "ridge", "lasso", or "ols".
#' @param alpha              Regularisation strength for ridge / lasso.
#' @param positive_coefficients Constrain coefficients to be non-negative.
#' @param output_csv         Optional path for the contributions CSV.
#'
#' @return A list with class \code{"MarketMixConfig"}.
#' @export
MarketMixConfig <- function(analysis_name = "MMM",
                            col_date = "date",
                            col_response = "revenue",
                            media_columns = character(),
                            control_columns = character(),
                            frequency = c("weekly", "daily", "monthly"),
                            default_decay_rate = 0.5,
                            default_saturation = c("log", "hill", "none"),
                            hill_half_max = 1.0,
                            hill_steepness = 1.0,
                            include_trend = TRUE,
                            include_seasonality = TRUE,
                            seasonality_period = 52L,
                            model_type = c("ridge", "lasso", "ols"),
                            alpha = 1.0,
                            positive_coefficients = TRUE,
                            output_csv = "") {
  frequency          <- match.arg(frequency)
  default_saturation <- match.arg(default_saturation)
  model_type         <- match.arg(model_type)

  cfg <- list(
    analysis_name         = analysis_name,
    col_date              = col_date,
    col_response          = col_response,
    media_columns         = as.character(media_columns),
    control_columns       = as.character(control_columns),
    frequency             = frequency,
    default_decay_rate    = default_decay_rate,
    default_saturation    = default_saturation,
    hill_half_max         = hill_half_max,
    hill_steepness        = hill_steepness,
    include_trend         = isTRUE(include_trend),
    include_seasonality   = isTRUE(include_seasonality),
    seasonality_period    = as.integer(seasonality_period),
    model_type            = model_type,
    alpha                 = alpha,
    positive_coefficients = isTRUE(positive_coefficients),
    output_csv            = output_csv
  )
  class(cfg) <- c("MarketMixConfig", "list")
  cfg
}


# ---- transformations -----------------------------------------------------

#' Apply geometric adstock to a numeric vector.
#' @keywords internal
.adstock <- function(x, decay) {
  out <- numeric(length(x))
  out[1] <- x[1]
  if (length(x) > 1) {
    for (t in seq.int(2, length(x))) {
      out[t] <- x[t] + decay * out[t - 1]
    }
  }
  out
}

#' Apply saturation transform.
#' @keywords internal
.saturate <- function(x, type, half_max = 1.0, steepness = 1.0) {
  switch(type,
         log  = log1p(abs(x)),
         hill = {
           num <- abs(x) ^ steepness
           num / (num + (half_max ^ steepness))
         },
         none = x,
         stop(sprintf("Unknown saturation '%s'", type)))
}

#' Build trend + Fourier seasonality features.
#' @keywords internal
.trend_seasonality <- function(n, include_trend, include_seasonality, period) {
  out <- list()
  if (include_trend) out$trend <- seq(0, 1, length.out = n)
  if (include_seasonality && period >= 2) {
    t <- seq_len(n) - 1
    out[[paste0("sin_", period)]] <- sin(2 * pi * t / period)
    out[[paste0("cos_", period)]] <- cos(2 * pi * t / period)
  }
  if (length(out) == 0) return(matrix(numeric(0), n, 0))
  do.call(cbind, out)
}


#' Market Mix Model pipeline (R6 class).
#'
#' Stages: load -> transform (adstock + saturation + structural) -> fit
#' (ridge / lasso via glmnet, or OLS) -> contribution decomposition + ROI +
#' response curves -> optional CSV write.
#'
#' @section Methods:
#' \describe{
#'   \item{\code{$run(cfg, dataframes)}}{Run the full pipeline. Returns a list
#'         with \code{contributions}, \code{channel_summary},
#'         \code{model_diagnostics}, \code{response_curves}, and \code{coefficients}.}
#' }
#'
#' @export
MarketMixPipeline <- R6::R6Class(
  "MarketMixPipeline",
  public = list(
    name        = "market_mix",
    description = "Adstock + saturation + ridge regression",
    last_output = NULL,

    initialize = function() invisible(self),

    run = function(cfg, dataframes) {
      stopifnot(inherits(cfg, "MarketMixConfig"))
      if (is.null(dataframes) || !"time_series" %in% names(dataframes)) {
        stop("dataframes must contain `time_series` for the R port.")
      }
      ts <- as.data.frame(dataframes$time_series)

      missing_cols <- setdiff(
        c(cfg$col_date, cfg$col_response, cfg$media_columns, cfg$control_columns),
        colnames(ts)
      )
      if (length(missing_cols)) {
        stop(sprintf("Missing columns: %s", paste(missing_cols, collapse = ", ")))
      }
      ts <- ts[order(ts[[cfg$col_date]]), , drop = FALSE]
      n <- nrow(ts)
      message(sprintf("Loaded %d periods", n))

      # ---- transform ----
      media_mat <- vapply(cfg$media_columns, function(col) {
        ad  <- .adstock(as.numeric(ts[[col]]), cfg$default_decay_rate)
        .saturate(ad, cfg$default_saturation,
                  half_max = cfg$hill_half_max,
                  steepness = cfg$hill_steepness)
      }, numeric(n))
      colnames(media_mat) <- paste0(cfg$media_columns, "_transformed")

      ctrl_mat <- if (length(cfg$control_columns)) {
        vapply(cfg$control_columns, function(col) as.numeric(ts[[col]]),
               numeric(n))
      } else matrix(numeric(0), n, 0)
      if (length(cfg$control_columns)) colnames(ctrl_mat) <- cfg$control_columns

      struct_mat <- .trend_seasonality(n, cfg$include_trend,
                                       cfg$include_seasonality,
                                       cfg$seasonality_period)

      X <- cbind(media_mat, ctrl_mat, struct_mat)
      y <- as.numeric(ts[[cfg$col_response]])

      # ---- fit ----
      fit_res <- private$.fit(X, y, cfg)
      coefs    <- fit_res$coef
      intercept <- fit_res$intercept
      y_pred   <- fit_res$y_pred

      r2 <- 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
      adj_r2 <- if (n > ncol(X) + 1)
        1 - (1 - r2) * (n - 1) / (n - ncol(X) - 1) else r2
      mape <- mean(abs((y[y != 0] - y_pred[y != 0]) / y[y != 0])) * 100

      # ---- contributions ----
      contrib <- as.data.frame(sweep(X, 2, coefs, "*"))
      contrib$base      <- intercept
      contrib$predicted <- y_pred
      contrib$actual    <- y
      contrib[[cfg$col_date]] <- ts[[cfg$col_date]]

      total_y <- sum(y)
      contribution_pct <- vapply(colnames(X), function(c) {
        if (total_y == 0) 0 else 100 * sum(contrib[[c]]) / total_y
      }, numeric(1))
      base_pct <- if (total_y == 0) 0 else 100 * intercept * n / total_y

      # ---- channel summary + ROI ----
      summary_rows <- lapply(cfg$media_columns, function(col) {
        feat <- paste0(col, "_transformed")
        spend <- sum(ts[[col]])
        contrib_total <- sum(contrib[[feat]])
        list(channel          = col,
             total_spend      = spend,
             total_contribution = contrib_total,
             contribution_pct = unname(contribution_pct[feat]),
             roi              = if (spend == 0) 0 else contrib_total / spend,
             coefficient      = coefs[[feat]],
             decay_rate       = cfg$default_decay_rate)
      })
      summary_rows[[length(summary_rows) + 1]] <- list(
        channel = "base", total_spend = 0, total_contribution = intercept * n,
        contribution_pct = base_pct, roi = 0, coefficient = intercept,
        decay_rate = 0
      )
      for (col in cfg$control_columns) {
        summary_rows[[length(summary_rows) + 1]] <- list(
          channel = col, total_spend = 0,
          total_contribution = sum(contrib[[col]]),
          contribution_pct = unname(contribution_pct[col]),
          roi = 0, coefficient = coefs[[col]], decay_rate = 0
        )
      }
      channel_summary <- .records_to_tibble(summary_rows)

      # ---- response curves ----
      response_curves <- do.call(rbind, lapply(cfg$media_columns, function(col) {
        feat <- paste0(col, "_transformed")
        spend_max <- max(ts[[col]], na.rm = TRUE) * 2
        spend_grid <- seq(0, spend_max, length.out = 50)
        sat_vals <- .saturate(spend_grid, cfg$default_saturation,
                              half_max = cfg$hill_half_max,
                              steepness = cfg$hill_steepness)
        data.frame(channel = col,
                   spend_level = spend_grid,
                   response = unname(coefs[[feat]]) * sat_vals,
                   stringsAsFactors = FALSE)
      }))

      diagnostics <- tibble::tibble(
        model_type = cfg$model_type, alpha = cfg$alpha,
        r_squared = r2, adjusted_r_squared = adj_r2,
        mape = mape, n_periods = n,
        n_features = ncol(X), intercept = intercept
      )

      out <- list(
        coefficients      = c(intercept = intercept, coefs),
        contributions     = tibble::as_tibble(contrib),
        channel_summary   = channel_summary,
        model_diagnostics = diagnostics,
        response_curves   = tibble::as_tibble(response_curves)
      )
      class(out) <- c("MarketMixOutput", "list")
      self$last_output <- out

      message(sprintf("R^2=%.4f  adj.R^2=%.4f  MAPE=%.2f%%", r2, adj_r2, mape))

      if (nzchar(cfg$output_csv)) {
        .ensure_parent(cfg$output_csv)
        utils::write.csv(out$contributions, cfg$output_csv, row.names = FALSE)
        message(sprintf("Wrote CSV: %s", cfg$output_csv))
      }
      invisible(out)
    }
  ),
  private = list(
    .fit = function(X, y, cfg) {
      n <- nrow(X)
      if (cfg$model_type == "ols" || n < ncol(X) + 5) {
        if (cfg$positive_coefficients) {
          # NNLS-equivalent: tiny ridge with non-negativity via glmnet
          fit <- glmnet::glmnet(X, y, alpha = 0, lambda = 1e-6 / n,
                                lower.limits = 0, intercept = TRUE,
                                standardize = FALSE)
          beta <- as.numeric(stats::coef(fit, s = 1e-6 / n))
          intercept <- beta[1]
          coefs <- stats::setNames(beta[-1], colnames(X))
        } else {
          df <- as.data.frame(X)
          df$.y <- y
          mod <- stats::lm(.y ~ ., data = df)
          intercept <- unname(stats::coef(mod)[1])
          coefs <- stats::setNames(stats::coef(mod)[-1], colnames(X))
          coefs[is.na(coefs)] <- 0
        }
      } else {
        gn_alpha <- if (cfg$model_type == "lasso") 1 else 0
        lower    <- if (cfg$positive_coefficients) 0 else -Inf
        # sklearn parametrises Ridge as RSS + alpha*||b||^2.
        # glmnet parametrises as (1/(2N))*RSS + lambda*((1-a)/2*||b||^2 + a*||b||_1).
        # For ridge (gn_alpha=0): glmnet_lambda = sklearn_alpha / N.
        # For lasso (gn_alpha=1): sklearn lasso uses 0.5*RSS/N + alpha*||b||_1
        #                         which equals glmnet_lambda = sklearn_alpha.
        gn_lambda <- if (cfg$model_type == "lasso") cfg$alpha else cfg$alpha / n
        fit <- glmnet::glmnet(X, y,
                              alpha = gn_alpha, lambda = gn_lambda,
                              lower.limits = lower, intercept = TRUE,
                              standardize = FALSE)
        beta <- as.numeric(stats::coef(fit, s = gn_lambda))
        intercept <- beta[1]
        coefs <- stats::setNames(beta[-1], colnames(X))
      }
      y_pred <- as.numeric(X %*% coefs + intercept)
      list(intercept = intercept, coef = coefs, y_pred = y_pred)
    }
  )
)
