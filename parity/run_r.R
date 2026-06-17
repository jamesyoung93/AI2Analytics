## Run the canonical Market Mix demo in R and save outputs as JSON.
## Run after parity/build_shared_data.py.
##
##   Rscript parity/run_r.R

suppressPackageStartupMessages({
  library(ai2analytics)
  library(jsonlite)
})

here <- normalizePath("parity")
ts <- read.csv(file.path(here, "shared_mmm.csv"))
ts$WEEK_ENDING <- as.Date(ts$WEEK_ENDING)

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
  positive_coefficients = TRUE,
  output_csv         = file.path(here, "_r_contribs.csv")
)

out <- suppressMessages(
  MarketMixPipeline$new()$run(cfg, dataframes = list(time_series = ts))
)

# Strip the leading 'intercept' element from coefficients
coefs <- out$coefficients
intercept <- unname(coefs["intercept"])
media_coefs   <- coefs[paste0(cfg$media_columns, "_transformed")]
control_coefs <- coefs[cfg$control_columns]

s <- out$channel_summary
get_channel <- function(field, ch) s[[field]][s$channel == ch]

payload <- list(
  language       = "r",
  n_periods      = as.integer(out$model_diagnostics$n_periods),
  r_squared      = unname(out$model_diagnostics$r_squared),
  adj_r_squared  = unname(out$model_diagnostics$adjusted_r_squared),
  mape           = unname(out$model_diagnostics$mape),
  intercept      = intercept,
  coefficients   = stats::setNames(as.list(unname(media_coefs)), cfg$media_columns),
  control_coefficients = stats::setNames(as.list(unname(control_coefs)), cfg$control_columns),
  channel_total_contribution = stats::setNames(
    as.list(vapply(cfg$media_columns,
                   function(ch) get_channel("total_contribution", ch),
                   numeric(1))),
    cfg$media_columns),
  channel_roi    = stats::setNames(
    as.list(vapply(cfg$media_columns,
                   function(ch) get_channel("roi", ch),
                   numeric(1))),
    cfg$media_columns),
  predicted_sum  = sum(out$contributions$predicted),
  actual_sum     = sum(out$contributions$actual),
  predicted_first5 = head(out$contributions$predicted, 5),
  predicted_last5  = tail(out$contributions$predicted, 5)
)

out_path <- file.path(here, "r_outputs.json")
write(toJSON(payload, auto_unbox = TRUE, pretty = TRUE, digits = 8), out_path)
cat("Wrote", out_path, "\n")
cat(substr(toJSON(payload, auto_unbox = TRUE, pretty = TRUE, digits = 6),
           1, 1000), "\n")
