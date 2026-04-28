test_that("DecisionStore round-trips records", {
  tmp <- tempfile(fileext = ".jsonl")
  ds <- DecisionStore$new(tmp)
  rid <- ds$log(decision_record(
    template_name = "segmentation",
    config_dict = list(col_entity_id = "npi", n_segments = 4),
    outcome_metrics = list(silhouette_score = 0.5),
    tags = c("us", "hcp")
  ))
  expect_true(nzchar(rid))
  expect_true(file.exists(tmp))

  records <- ds$query()
  expect_length(records, 1)
  expect_equal(records[[1]]$template_name, "segmentation")
  expect_equal(records[[1]]$run_id, rid)

  # tag filtering
  expect_length(ds$query(tags = "us"), 1)
  expect_length(ds$query(tags = "eu"), 0)
})

test_that("ContextStore filters by scope, category, template", {
  tmp <- tempfile(fileext = ".jsonl")
  cs <- ContextStore$new(tmp)
  cs$add(context_entry(scope = list(region = "us"), category = "column_mapping",
                       title = "US uses npi", template_name = "segmentation",
                       confidence = 0.9))
  cs$add(context_entry(scope = list(region = "eu"), category = "column_mapping",
                       title = "EU uses PRESCRIBER_ID", template_name = "segmentation",
                       confidence = 0.85))
  cs$add(context_entry(scope = list(), category = "data_quality",
                       title = "Beware NA in TIER", template_name = "segmentation",
                       confidence = 0.6))

  expect_length(cs$query(), 3)
  expect_length(cs$query(scope = list(region = "us")), 2)  # us + scope-less entries
  expect_length(cs$query(category = "data_quality"), 1)
  # Highest confidence first
  q <- cs$query(category = "column_mapping")
  expect_equal(q[[1]]$title, "US uses npi")
})

test_that("KnowledgeRetriever produces non-empty block when stores have data", {
  tmpd <- tempfile(fileext = ".jsonl")
  tmpc <- tempfile(fileext = ".jsonl")
  ds <- DecisionStore$new(tmpd)
  cs <- ContextStore$new(tmpc)
  ds$log(decision_record(template_name = "segmentation",
                         config_dict = list(col_entity_id = "npi"),
                         tags = c("us")))
  cs$add(context_entry(scope = list(region = "us"),
                       category = "column_mapping",
                       title = "US uses npi", confidence = 0.9))

  ret <- KnowledgeRetriever$new(decision_store = ds, context_store = cs)
  blk <- ret$retrieve(template_name = "segmentation",
                      scope = list(region = "us"))
  expect_true(grepl("KNOWLEDGE BASE", blk))
  expect_true(grepl("npi", blk))
})
