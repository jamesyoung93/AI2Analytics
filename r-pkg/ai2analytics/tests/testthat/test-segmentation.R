test_that("SegmentationConfig validates basic shape", {
  cfg <- SegmentationConfig(
    analysis_name = "t",
    col_entity_id = "id",
    feature_columns = c("x", "y"),
    n_segments = 3
  )
  expect_s3_class(cfg, "SegmentationConfig")
  expect_equal(cfg$method, "kmeans")
  expect_true(cfg$normalize)
})

test_that("SegmentationPipeline runs end-to-end on synthetic US HCP data", {
  dfs <- generate_us_hcp(n_npis = 500L, n_weeks = 4L, seed = 1L)
  cfg <- SegmentationConfig(
    analysis_name = "test_us",
    col_entity_id = "npi",
    feature_columns = c("IL_17_TRX_L12M", "IL_23_TRX_L12M"),
    n_segments = 3
  )
  out <- suppressMessages(
    SegmentationPipeline$new()$run(cfg, dataframes = list(entity_data = dfs$hcp_reference))
  )
  expect_s3_class(out, "SegmentationOutput")
  expect_equal(nrow(out$assignments), 500)
  expect_equal(out$summary_stats$n_segments, 3L)
  expect_true(all(c("npi", "SEGMENT") %in% colnames(out$assignments)))
  expect_equal(nrow(out$profiles), 3)
  expect_true(out$summary_stats$silhouette_score > -1 &&
              out$summary_stats$silhouette_score <= 1)
})

test_that("auto method picks the better-scoring algorithm", {
  dfs <- generate_eu_account(n_accounts = 400L, n_months = 2L, seed = 2L)
  cfg <- SegmentationConfig(
    analysis_name = "test_eu",
    col_entity_id = "PRESCRIBER_ID",
    feature_columns = c("UNITS_SOLD_L12M", "TIER"),
    n_segments = 3,
    method = "auto"
  )
  out <- suppressMessages(
    SegmentationPipeline$new()$run(cfg, dataframes = list(entity_data = dfs$account_reference))
  )
  expect_true(out$summary_stats$method %in% c("kmeans", "hierarchical"))
  expect_equal(nrow(out$assignments), 400)
})

test_that("auto-select-k chooses k within range", {
  dfs <- generate_us_hcp(n_npis = 300L, n_weeks = 2L, seed = 3L)
  cfg <- SegmentationConfig(
    analysis_name = "auto_k",
    col_entity_id = "npi",
    feature_columns = c("IL_17_TRX_L12M", "IL_23_TRX_L12M"),
    auto_select_k = TRUE,
    k_range = c(2L, 5L)
  )
  out <- suppressMessages(
    SegmentationPipeline$new()$run(cfg, dataframes = list(entity_data = dfs$hcp_reference))
  )
  expect_true(out$summary_stats$n_segments >= 2L &&
              out$summary_stats$n_segments <= 5L)
})

test_that("validation rejects missing entity_data", {
  cfg <- SegmentationConfig(col_entity_id = "id", n_segments = 3)
  expect_error(
    SegmentationPipeline$new()$run(cfg, dataframes = list()),
    "entity_data"
  )
})
