make_synthetic_mmm <- function(n = 104, seed = 7) {
  set.seed(seed)
  dates <- seq.Date(from = as.Date("2024-01-05"), by = "week", length.out = n)
  trend <- seq(100, 150, length.out = n)
  season <- 20 * sin(2 * pi * seq_len(n) / 52)
  tv  <- pmin(200, stats::rexp(n, rate = 1 / 50))
  dig <- pmin(150, stats::rexp(n, rate = 1 / 30))
  pr  <- pmin(100, stats::rexp(n, rate = 1 / 20))
  sales <- trend + season +
    0.8 * log1p(tv)  * 10 +
    0.5 * log1p(dig) * 8 +
    0.3 * log1p(pr)  * 5 +
    stats::rnorm(n, 0, 8)
  data.frame(
    WEEK_ENDING      = dates,
    SALES            = round(sales),
    TV_SPEND         = round(tv, 2),
    DIGITAL_SPEND    = round(dig, 2),
    PRINT_SPEND      = round(pr, 2),
    PRICE_INDEX      = 100 + stats::rnorm(n, 0, 2),
    DISTRIBUTION_PCT = pmin(100, pmax(70, 85 + stats::rnorm(n, 0, 3)))
  )
}

test_that("MarketMixPipeline fits and decomposes contributions", {
  ts <- make_synthetic_mmm()
  cfg <- MarketMixConfig(
    analysis_name = "tmmm",
    col_date = "WEEK_ENDING",
    col_response = "SALES",
    media_columns = c("TV_SPEND", "DIGITAL_SPEND", "PRINT_SPEND"),
    control_columns = c("PRICE_INDEX", "DISTRIBUTION_PCT"),
    default_saturation = "log"
  )
  out <- suppressMessages(
    MarketMixPipeline$new()$run(cfg, dataframes = list(time_series = ts))
  )
  expect_s3_class(out, "MarketMixOutput")
  expect_true(out$model_diagnostics$r_squared > 0.3)
  expect_true(all(c("TV_SPEND", "DIGITAL_SPEND", "PRINT_SPEND", "base") %in%
                  out$channel_summary$channel))
  # Contribution percentages should be reasonable
  pct <- sum(out$channel_summary$contribution_pct)
  expect_true(pct > 50 && pct < 200)
  # Response curves: 50 grid points x 3 channels
  expect_equal(nrow(out$response_curves), 150)
})

test_that("MarketMixPipeline handles ridge with positive coefficients", {
  ts <- make_synthetic_mmm()
  cfg <- MarketMixConfig(
    analysis_name = "tmmm_ridge",
    col_date = "WEEK_ENDING",
    col_response = "SALES",
    media_columns = c("TV_SPEND", "DIGITAL_SPEND"),
    model_type = "ridge",
    alpha = 0.5,
    positive_coefficients = TRUE
  )
  out <- suppressMessages(
    MarketMixPipeline$new()$run(cfg, dataframes = list(time_series = ts))
  )
  # Drop intercept and check no negatives among media coefficients
  media_coefs <- out$coefficients[grepl("_transformed$", names(out$coefficients))]
  expect_true(all(media_coefs >= 0))
})

test_that("validation rejects missing time_series", {
  cfg <- MarketMixConfig(media_columns = "TV_SPEND")
  expect_error(MarketMixPipeline$new()$run(cfg, dataframes = list()),
               "time_series")
})
