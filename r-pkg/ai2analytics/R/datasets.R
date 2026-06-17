## Synthetic pharma datasets, ported from Python `ai2analytics.datasets`.
## All identifiers are random -- no real PII. Distributions chosen to mimic
## the Python generators reasonably; not bit-identical.

#' Generate synthetic US HCP pharma data.
#'
#' Produces seven tables matching the schema expected by the detail-optimization
#' template: \code{hcp_reference}, \code{hcp_weekly}, \code{calls},
#' \code{team_a_alignment}, \code{team_b_alignment}, \code{portfolio_decile},
#' \code{priority_targets}.
#'
#' @param n_npis      Number of synthetic providers.
#' @param n_weeks     Number of consecutive weeks for the weekly tables.
#' @param writer_rate Fraction of HCPs that are writers.
#' @param target_rate Fraction of HCPs that are promotional targets.
#' @param seed        Integer seed for reproducibility.
#' @return Named list of tibbles.
#' @export
generate_us_hcp <- function(n_npis = 5000L, n_weeks = 52L,
                            writer_rate = 0.35, target_rate = 0.60,
                            seed = 42L) {
  withr::with_seed(seed, {
    npis <- sample(1e9:9.999999999e9, n_npis, replace = FALSE)
    is_writer <- stats::runif(n_npis) < writer_rate
    is_target <- stats::runif(n_npis) < target_rate
    il17 <- ifelse(is_writer,
                   stats::rnbinom(n_npis, size = 3, prob = 0.15),
                   stats::rnbinom(n_npis, size = 1, prob = 0.40))
    il23 <- ifelse(is_writer,
                   stats::rnbinom(n_npis, size = 2, prob = 0.20),
                   stats::rnbinom(n_npis, size = 1, prob = 0.50))

    hcp_reference <- tibble::tibble(
      npi = as.numeric(npis),
      WRITER_FLAG = ifelse(is_writer, "Y", "N"),
      TARGET_FLAG = ifelse(is_target, "Y", "N"),
      IL_17_TRX_L12M = as.integer(il17),
      IL_23_TRX_L12M = as.integer(il23)
    )

    fridays <- seq.Date(from = as.Date("2025-01-03"), by = "week",
                        length.out = n_weeks)
    seasonal <- 1 + 0.25 * cos(2 * pi * as.integer(format(fridays, "%j")) / 365.25)

    weekly_records <- list()
    indications <- c("IND_A", "IND_B", "IND_C", "IND_D")
    ind_weights <- c(0.40, 0.30, 0.20, 0.10)
    for (i in seq_len(n_weeks)) {
      obs_p <- ifelse(is_writer, 0.7, 0.3)
      observed <- which(stats::runif(n_npis) < obs_p)
      if (length(observed) == 0) next
      sub_npis <- hcp_reference$npi[observed]
      sub_writer <- is_writer[observed]
      lam <- ifelse(sub_writer, 2.5, 0.4) * seasonal[i]
      zero_inflate_p <- ifelse(sub_writer, 0.20, 0.55)
      zinf <- stats::runif(length(observed)) < zero_inflate_p
      referrals <- ifelse(zinf, 0L, stats::rpois(length(observed), lam))
      weekly_records[[i]] <- data.frame(
        npi = sub_npis,
        WEEK_ENDING = fridays[i],
        INDC = sample(indications, length(observed), replace = TRUE,
                      prob = ind_weights),
        PAT_COUNT_REFERRED = as.integer(referrals),
        TARGET_FLAG = ifelse(is_target[observed], "Y", "N"),
        SYMPTOM_SEVERITY_SCORE = round(stats::rlnorm(length(observed), 1.0, 0.6), 2),
        PRIOR_AUTH_COUNT = stats::rpois(length(observed),
                                        ifelse(sub_writer, 0.8, 0.3)),
        SWITCH_FLAG_NUM = stats::rbinom(length(observed), 1,
                                        ifelse(sub_writer, 0.12, 0.04)),
        stringsAsFactors = FALSE
      )
    }
    hcp_weekly <- tibble::as_tibble(do.call(rbind, weekly_records))

    # Calls: targets get more
    target_lookup <- stats::setNames(is_target, hcp_reference$npi)
    call_records <- list()
    for (npi in hcp_reference$npi) {
      is_tgt <- target_lookup[as.character(npi)]
      cp <- if (is_tgt) 0.25 else 0.08
      mask <- stats::runif(n_weeks) < cp
      if (!any(mask)) next
      n_calls <- stats::rnbinom(sum(mask), size = 2, prob = 0.55)
      keep <- n_calls > 0
      if (!any(keep)) next
      call_records[[length(call_records) + 1]] <- data.frame(
        NPI = npi,
        WEEK_ENDING = fridays[which(mask)[keep]],
        HCP_F2F_CALLS = as.integer(n_calls[keep]),
        stringsAsFactors = FALSE
      )
    }
    calls <- tibble::as_tibble(do.call(rbind, call_records))

    # Team alignments (some HCPs uncovered)
    align <- function(npis, n_terr, coverage) {
      mask <- stats::runif(length(npis)) < coverage
      tibble::tibble(HCP_NPI = npis[mask],
                     TERRITORY_ID = sample.int(n_terr, sum(mask),
                                               replace = TRUE))
    }
    team_a <- align(hcp_reference$npi, 50L, 0.85)
    team_b <- align(hcp_reference$npi, 40L, 0.80)

    # Portfolio decile correlated with writer status
    raw <- ifelse(is_writer,
                  stats::rbeta(n_npis, 5, 2),
                  stats::rbeta(n_npis, 2, 5))
    portfolio <- tibble::tibble(
      npi = hcp_reference$npi,
      PORTFOLIO_UNITS_DECILE = pmax(1L, pmin(10L, ceiling(raw * 10)))
    )

    # Priority: top writers
    writers <- hcp_reference[hcp_reference$WRITER_FLAG == "Y", ]
    if (nrow(writers) > 0) {
      score <- writers$IL_17_TRX_L12M + writers$IL_23_TRX_L12M + 1e-6
      n_pri <- max(1L, min(round(0.10 * nrow(hcp_reference)), nrow(writers)))
      idx <- sample.int(nrow(writers), n_pri, prob = score / sum(score),
                        replace = FALSE)
      priority <- tibble::tibble(npi = writers$npi[idx], PRIORITY_TARGET = "Y")
    } else {
      priority <- tibble::tibble(npi = numeric(0), PRIORITY_TARGET = character(0))
    }

    list(
      hcp_reference    = hcp_reference,
      hcp_weekly       = hcp_weekly,
      calls            = calls,
      team_a_alignment = team_a,
      team_b_alignment = team_b,
      portfolio_decile = portfolio,
      priority_targets = priority
    )
  })
}


#' Generate synthetic EU account-level pharma data.
#'
#' @param n_accounts Number of accounts.
#' @param n_months   Number of months of monthly history.
#' @param seed       Integer seed.
#' @return Named list of tibbles.
#' @export
generate_eu_account <- function(n_accounts = 3000L, n_months = 24L,
                                seed = 43L) {
  withr::with_seed(seed, {
    ids <- sprintf("ACC-%05d", sample.int(99999L, n_accounts))
    regions <- sample(c("UK", "DE", "FR", "IT", "ES", "NL"),
                      n_accounts, replace = TRUE)
    segments <- sample(c("A", "B", "C"), n_accounts, replace = TRUE,
                       prob = c(0.2, 0.5, 0.3))
    tier <- sample(1:4, n_accounts, replace = TRUE,
                   prob = c(0.15, 0.30, 0.35, 0.20))
    # Volume correlates with tier
    units_l12m <- pmax(0L, as.integer(round(stats::rlnorm(n_accounts,
      meanlog = 6 - 0.4 * tier, sdlog = 0.7))))
    name_pre <- sample(c("Hospital", "Clinic", "MedCenter", "Pharmacy"),
                       n_accounts, replace = TRUE)
    name_suf <- sample(c("North", "South", "East", "West", "Central"),
                       n_accounts, replace = TRUE)
    account_reference <- tibble::tibble(
      PRESCRIBER_ID  = ids,
      ACCOUNT_NAME   = paste(name_pre, name_suf),
      REGION         = regions,
      SEGMENT        = segments,
      TIER           = as.integer(tier),
      UNITS_SOLD_L12M = units_l12m
    )

    months <- seq.Date(from = as.Date("2024-01-31"), by = "month",
                       length.out = n_months)
    monthly_records <- list()
    for (j in seq_len(n_months)) {
      base <- units_l12m / 12
      noise <- stats::rnorm(n_accounts, mean = 1, sd = 0.2)
      units <- pmax(0L, as.integer(round(base * noise)))
      monthly_records[[j]] <- tibble::tibble(
        PRESCRIBER_ID = ids,
        MONTH_END     = months[j],
        UNITS_SOLD    = units,
        NEW_PATIENTS  = stats::rpois(n_accounts, lambda = units / 4),
        MARKET_SHARE  = pmin(1, pmax(0, stats::rbeta(n_accounts, 2, 4)))
      )
    }
    account_monthly <- do.call(rbind, monthly_records)

    visit_records <- list()
    for (j in seq_len(n_months)) {
      kp <- ifelse(tier <= 2, 0.7, 0.3)
      mask <- stats::runif(n_accounts) < kp
      if (!any(mask)) next
      visit_records[[length(visit_records) + 1]] <- tibble::tibble(
        PRESCRIBER_ID    = ids[mask],
        MONTH_END        = months[j],
        KAM_VISITS       = stats::rpois(sum(mask), 1.5),
        MEDICAL_VISITS   = stats::rpois(sum(mask), 0.5),
        DIGITAL_CONTACTS = stats::rpois(sum(mask), 3.0)
      )
    }
    visits <- do.call(rbind, visit_records)

    kam_align <- tibble::tibble(
      PRESCRIBER_ID    = ids,
      KAM_TERRITORY_ID = sample.int(60L, n_accounts, replace = TRUE)
    )
    medical_align <- tibble::tibble(
      PRESCRIBER_ID        = ids,
      MEDICAL_TERRITORY_ID = sample.int(30L, n_accounts, replace = TRUE)
    )

    list(
      account_reference = account_reference,
      account_monthly   = account_monthly,
      visits            = visits,
      kam_alignment     = kam_align,
      medical_alignment = medical_align
    )
  })
}


#' Generate synthetic BRIC quarterly pharma data.
#'
#' @param n_accounts Number of accounts.
#' @param n_quarters Number of quarters of history.
#' @param seed       Integer seed.
#' @return Named list of tibbles.
#' @export
generate_bric <- function(n_accounts = 4000L, n_quarters = 12L, seed = 44L) {
  withr::with_seed(seed, {
    ids <- as.integer(seq_len(n_accounts) + 100000L)
    cc  <- sample(c("BR", "RU", "IN", "CN"), n_accounts, replace = TRUE)
    cities <- sample(c("Mumbai", "Beijing", "Shanghai", "Sao Paulo",
                       "Moscow", "Delhi", "Rio", "Guangzhou"),
                     n_accounts, replace = TRUE)
    inst_type <- sample(c("hospital", "clinic", "pharmacy"),
                        n_accounts, replace = TRUE,
                        prob = c(0.3, 0.5, 0.2))
    channel <- ifelse(inst_type == "hospital", "hospital", "retail")
    customer <- sample(c("A", "B", "C", "D"), n_accounts, replace = TRUE,
                       prob = c(0.1, 0.25, 0.4, 0.25))

    account_master <- tibble::tibble(
      ACCOUNT_ID       = ids,
      COUNTRY_CODE     = cc,
      CITY             = cities,
      INSTITUTION_TYPE = inst_type,
      CHANNEL          = channel,
      CUSTOMER_SEGMENT = customer
    )

    quarters <- seq.Date(from = as.Date("2023-03-31"), by = "quarter",
                         length.out = n_quarters)
    perf_records <- list()
    for (q in seq_len(n_quarters)) {
      mask <- stats::runif(n_accounts) < 0.85
      n <- sum(mask)
      perf_records[[q]] <- tibble::tibble(
        ACCOUNT_ID       = ids[mask],
        QUARTER_END      = quarters[q],
        REVENUE_LOCAL    = round(stats::rlnorm(n, 9, 1.2), 2),
        UNITS_DISPENSED  = stats::rpois(n, lambda = 200),
        PATIENT_STARTS   = stats::rpois(n, lambda = 8),
        COMPLIANCE_RATE  = pmin(1, pmax(0, stats::rbeta(n, 8, 3)))
      )
    }
    quarterly_performance <- do.call(rbind, perf_records)

    eng_records <- list()
    for (q in seq_len(n_quarters)) {
      mask <- stats::runif(n_accounts) < 0.6
      n <- sum(mask)
      eng_records[[q]] <- tibble::tibble(
        ACCOUNT_ID               = ids[mask],
        QUARTER_END              = quarters[q],
        REP_VISITS               = stats::rnbinom(n, size = 2, prob = 0.4),
        CONGRESS_ATTENDANCE      = stats::rbinom(n, 1, 0.2),
        SAMPLE_UNITS             = stats::rpois(n, lambda = 30),
        DIGITAL_ENGAGEMENT_SCORE = pmin(100L, as.integer(round(stats::rlnorm(n, 2.5, 0.7))))
      )
    }
    engagement <- do.call(rbind, eng_records)

    sales_alignment <- tibble::tibble(
      ACCOUNT_ID          = ids,
      SALES_TERRITORY_ID  = sample.int(80L, n_accounts, replace = TRUE),
      COUNTRY_CODE        = cc
    )

    list(
      account_master         = account_master,
      quarterly_performance  = quarterly_performance,
      engagement             = engagement,
      sales_alignment        = sales_alignment
    )
  })
}
