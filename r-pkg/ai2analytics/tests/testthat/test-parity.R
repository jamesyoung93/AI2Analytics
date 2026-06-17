## Regression test: pin the canonical Market Mix outputs against the values
## the Python implementation produces on the *same* CSV (parity/shared_mmm.csv).
## If anyone changes the R fitting logic and breaks parity, this catches it.

skip_if_no_parity_csv <- function() {
  testthat::skip_if_not(
    file.exists(file.path("..", "..", "..", "..", "parity", "shared_mmm.csv")) ||
      file.exists("../../parity/shared_mmm.csv") ||
      file.exists("parity/shared_mmm.csv"),
    "parity/shared_mmm.csv not present (run `python parity/build_shared_data.py`)"
  )
}

find_parity_csv <- function() {
  candidates <- c(
    file.path("..", "..", "..", "..", "parity", "shared_mmm.csv"),
    file.path("..", "..", "parity", "shared_mmm.csv"),
    file.path("parity", "shared_mmm.csv")
  )
  for (p in candidates) {
    if (file.exists(p)) return(normalizePath(p))
  }
  NULL
}

test_that("Canonical MMM run matches Python within tolerance", {
  skip_if_no_parity_csv()
  csv <- find_parity_csv()
  testthat::skip_if(is.null(csv))
  ts <- read.csv(csv); ts$WEEK_ENDING <- as.Date(ts$WEEK_ENDING)

  cfg <- MarketMixConfig(
    analysis_name      = "parity_mmm",
    col_date           = "WEEK_ENDING",
    col_response       = "SALES",
    media_columns      = c("TV_SPEND", "DIGITAL_SPEND", "PRINT_SPEND"),
    control_columns    = c("PRICE_INDEX", "DISTRIBUTION_PCT"),
    default_decay_rate = 0.5,
    default_saturation = "log",
    model_type         = "ridge",
    alpha              = 1.0,
    positive_coefficients = TRUE
  )
  out <- suppressMessages(MarketMixPipeline$new()$run(
    cfg, dataframes = list(time_series = ts)
  ))

  # These values were captured from the Python ai2analytics 0.2.0 reference run
  # with the same CSV (alpha=1.0, log saturation, ridge, positive coefs).
  py_r2 <- 0.7900
  py_roi_tv  <- 1.1037
  py_roi_dig <- 0.8037
  py_roi_pr  <- 0.2118

  expect_equal(unname(out$model_diagnostics$r_squared), py_r2, tolerance = 0.05)

  s <- out$channel_summary
  roi_tv  <- s$roi[s$channel == "TV_SPEND"]
  roi_dig <- s$roi[s$channel == "DIGITAL_SPEND"]
  roi_pr  <- s$roi[s$channel == "PRINT_SPEND"]
  # 20% relative agreement -- different solvers (sklearn Ridge vs glmnet)
  # cannot be bit-identical even with matched parameterisation.
  expect_lt(abs(roi_tv  - py_roi_tv)  / py_roi_tv,  0.20)
  expect_lt(abs(roi_dig - py_roi_dig) / py_roi_dig, 0.20)
  expect_lt(abs(roi_pr  - py_roi_pr)  / py_roi_pr,  0.20)

  # Ranking parity: TV > Digital > Print
  expect_true(roi_tv > roi_dig)
  expect_true(roi_dig > roi_pr)
})
