test_that("generate_us_hcp returns the expected tables", {
  d <- generate_us_hcp(n_npis = 200L, n_weeks = 3L, seed = 11L)
  expect_setequal(
    names(d),
    c("hcp_reference", "hcp_weekly", "calls",
      "team_a_alignment", "team_b_alignment",
      "portfolio_decile", "priority_targets")
  )
  expect_equal(nrow(d$hcp_reference), 200)
  expect_true(all(d$portfolio_decile$PORTFOLIO_UNITS_DECILE %in% 1:10))
  expect_true(all(d$hcp_reference$WRITER_FLAG %in% c("Y", "N")))
})

test_that("generate_eu_account returns expected tables", {
  d <- generate_eu_account(n_accounts = 100L, n_months = 3L, seed = 12L)
  expect_setequal(names(d),
                  c("account_reference", "account_monthly", "visits",
                    "kam_alignment", "medical_alignment"))
  expect_equal(nrow(d$account_reference), 100)
  expect_true(all(grepl("^ACC-", d$account_reference$PRESCRIBER_ID)))
  expect_true(all(d$account_reference$TIER %in% 1:4))
})

test_that("generate_bric returns expected tables", {
  d <- generate_bric(n_accounts = 200L, n_quarters = 4L, seed = 13L)
  expect_setequal(names(d),
                  c("account_master", "quarterly_performance",
                    "engagement", "sales_alignment"))
  expect_equal(nrow(d$account_master), 200)
  expect_true(all(d$account_master$COUNTRY_CODE %in%
                  c("BR", "RU", "IN", "CN")))
})

test_that("seeds are reproducible", {
  a <- generate_us_hcp(n_npis = 50L, n_weeks = 2L, seed = 99L)
  b <- generate_us_hcp(n_npis = 50L, n_weeks = 2L, seed = 99L)
  expect_equal(a$hcp_reference, b$hcp_reference)
})
