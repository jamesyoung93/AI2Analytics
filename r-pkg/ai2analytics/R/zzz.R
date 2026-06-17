#' ai2analytics: AI-Powered Last-Mile Analytics Framework
#'
#' Native R port of the Python AI2Analytics package. Provides reusable
#' pipeline templates that separate analytical logic from data shape, so the
#' same template runs across brands, regions, and schemas via configuration.
#'
#' @section Templates:
#' \itemize{
#'   \item \code{\link{SegmentationPipeline}}: K-means / hierarchical entity
#'         clustering with optional auto-k and method selection.
#'   \item \code{\link{MarketMixPipeline}}: Adstock + saturation + ridge
#'         regression with contribution decomposition and response curves.
#' }
#'
#' @section Knowledge:
#' \itemize{
#'   \item \code{\link{DecisionStore}}: JSONL-backed log of past pipeline runs.
#'   \item \code{\link{ContextStore}}: synthesized patterns and best practices.
#'   \item \code{\link{KnowledgeRetriever}}: format both for prompt injection.
#' }
#'
#' @section Synthetic data:
#' \code{\link{generate_us_hcp}}, \code{\link{generate_eu_account}},
#' \code{\link{generate_bric}} produce realistic fake datasets with no PII.
#'
#' @keywords internal
"_PACKAGE"

# Avoid R CMD check NOTEs for tidy-eval column names used in dplyr verbs.
utils::globalVariables(c(
  "SEGMENT", "npi", "WRITER_FLAG", "TARGET_FLAG", "WEEK_ENDING",
  "INDC", "PAT_COUNT_REFERRED", "SYMPTOM_SEVERITY_SCORE",
  "PRIOR_AUTH_COUNT", "SWITCH_FLAG_NUM", "HCP_NPI", "TERRITORY_ID",
  "PORTFOLIO_UNITS_DECILE", "PRIORITY_TARGET", "IL_17_TRX_L12M",
  "IL_23_TRX_L12M", "PRESCRIBER_ID", "ACCOUNT_NAME", "REGION",
  "SEGMENT", "TIER", "UNITS_SOLD_L12M", "MONTH_END", "UNITS_SOLD",
  "NEW_PATIENTS", "MARKET_SHARE", "KAM_VISITS", "MEDICAL_VISITS",
  "DIGITAL_CONTACTS", "KAM_TERRITORY_ID", "MEDICAL_TERRITORY_ID",
  "ACCOUNT_ID", "COUNTRY_CODE", "CITY", "INSTITUTION_TYPE", "CHANNEL",
  "CUSTOMER_SEGMENT", "QUARTER_END", "REVENUE_LOCAL", "UNITS_DISPENSED",
  "PATIENT_STARTS", "COMPLIANCE_RATE", "REP_VISITS",
  "CONGRESS_ATTENDANCE", "SAMPLE_UNITS", "DIGITAL_ENGAGEMENT_SCORE",
  "SALES_TERRITORY_ID"
))
