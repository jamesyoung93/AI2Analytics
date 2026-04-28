## Knowledge module: DecisionStore, ContextStore, KnowledgeRetriever.
## JSONL-backed only in the R port -- no Spark/Delta backend.

#' Build a decision record (immutable list).
#'
#' @param template_name   Template that produced this run.
#' @param config_dict     Named list of config values used.
#' @param data_profile    Free-text description of the input data.
#' @param user_answers    Named list of user-provided answers (if any).
#' @param auto_detected   Named list of auto-detected config values.
#' @param adapter_code    Adapter code string (if any).
#' @param outcome_notes   Free-text notes on the outcome.
#' @param outcome_metrics Named list of numeric outcome metrics.
#' @param tags            Character vector of tags.
#' @return A list with class \code{"DecisionRecord"}.
#' @export
decision_record <- function(template_name = "",
                            config_dict = list(),
                            data_profile = "",
                            user_answers = list(),
                            auto_detected = list(),
                            adapter_code = "",
                            outcome_notes = "",
                            outcome_metrics = list(),
                            tags = character()) {
  rec <- list(
    run_id          = "",
    timestamp       = "",
    template_name   = template_name,
    config_dict     = as.list(config_dict),
    data_profile    = data_profile,
    user_answers    = as.list(user_answers),
    auto_detected   = as.list(auto_detected),
    adapter_code    = adapter_code,
    outcome_notes   = outcome_notes,
    outcome_metrics = as.list(outcome_metrics),
    tags            = as.character(tags)
  )
  class(rec) <- c("DecisionRecord", "list")
  rec
}

#' Build a context entry.
#'
#' @param scope         Named character list (e.g. list(region = "us")).
#' @param category      One of "column_mapping", "data_quality",
#'                      "adapter_pattern", "config_preference", "troubleshooting".
#' @param title         Short title.
#' @param content       Detailed description.
#' @param template_name Optional template name.
#' @param confidence    Numeric in [0, 1].
#' @param source_run_ids Character vector of source run IDs.
#' @return A list with class \code{"ContextEntry"}.
#' @export
context_entry <- function(scope = list(),
                          category = "general",
                          title = "",
                          content = "",
                          template_name = "",
                          confidence = 0.5,
                          source_run_ids = character()) {
  ent <- list(
    entry_id       = "",
    created        = "",
    updated        = "",
    scope          = as.list(scope),
    category       = category,
    title          = title,
    content        = content,
    template_name  = template_name,
    confidence     = as.numeric(confidence),
    source_run_ids = as.character(source_run_ids)
  )
  class(ent) <- c("ContextEntry", "list")
  ent
}


# ---- helpers -------------------------------------------------------------

.gen_id <- function(n = 12) {
  paste(sample(c(0:9, letters[1:6]), n, replace = TRUE), collapse = "")
}

.now_iso <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%OS6Z", tz = "UTC")

.read_jsonl <- function(path) {
  if (!file.exists(path)) return(list())
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(lines)]
  lapply(lines, function(l) jsonlite::fromJSON(l, simplifyVector = FALSE))
}

.append_jsonl <- function(path, obj) {
  con <- file(path, open = "a", encoding = "UTF-8")
  on.exit(close(con))
  cat(jsonlite::toJSON(obj, auto_unbox = TRUE, null = "null"), file = con,
      sep = "\n")
}


#' Decision store: append-only JSONL log of pipeline runs.
#'
#' @section Methods:
#' \describe{
#'   \item{\code{$log(record)}}{Append a \code{decision_record()}; returns the run_id.}
#'   \item{\code{$query(template_name=NULL, tags=NULL, limit=10)}}{Return matching
#'         records, most recent first.}
#' }
#'
#' @export
DecisionStore <- R6::R6Class(
  "DecisionStore",
  public = list(
    path = NULL,
    initialize = function(path) {
      self$path <- path
      d <- dirname(path)
      if (nzchar(d) && d != "." && !dir.exists(d)) {
        dir.create(d, recursive = TRUE, showWarnings = FALSE)
      }
    },
    log = function(record) {
      stopifnot(inherits(record, "DecisionRecord"))
      record$run_id    <- if (nzchar(record$run_id)) record$run_id else .gen_id()
      record$timestamp <- if (nzchar(record$timestamp)) record$timestamp else .now_iso()
      .append_jsonl(self$path, record)
      record$run_id
    },
    query = function(template_name = NULL, tags = NULL, limit = 10L) {
      records <- .read_jsonl(self$path)
      if (!is.null(template_name)) {
        records <- Filter(function(r) identical(r$template_name, template_name),
                          records)
      }
      if (!is.null(tags) && length(tags)) {
        records <- Filter(function(r) length(intersect(unlist(r$tags), tags)) > 0,
                          records)
      }
      records <- records[order(vapply(records, function(r)
        as.character(r$timestamp), character(1)), decreasing = TRUE)]
      head(records, limit)
    }
  )
)

#' Context store: append-only JSONL log of synthesised context entries.
#'
#' @section Methods:
#' \describe{
#'   \item{\code{$add(entry)}}{Append a \code{context_entry()}; returns the entry_id.}
#'   \item{\code{$query(scope=NULL, category=NULL, template_name=NULL, limit=10)}}{
#'         Return matching entries, highest confidence first.}
#' }
#'
#' @export
ContextStore <- R6::R6Class(
  "ContextStore",
  public = list(
    path = NULL,
    initialize = function(path) {
      self$path <- path
      d <- dirname(path)
      if (nzchar(d) && d != "." && !dir.exists(d)) {
        dir.create(d, recursive = TRUE, showWarnings = FALSE)
      }
    },
    add = function(entry) {
      stopifnot(inherits(entry, "ContextEntry"))
      entry$entry_id <- if (nzchar(entry$entry_id)) entry$entry_id else .gen_id()
      now <- .now_iso()
      if (!nzchar(entry$created)) entry$created <- now
      entry$updated <- now
      .append_jsonl(self$path, entry)
      entry$entry_id
    },
    query = function(scope = NULL, category = NULL, template_name = NULL,
                     limit = 10L) {
      entries <- .read_jsonl(self$path)
      if (!is.null(scope)) {
        entries <- Filter(function(e) {
          es <- e$scope
          all(vapply(names(es), function(k) {
            identical(scope[[k]], es[[k]])
          }, logical(1)))
        }, entries)
      }
      if (!is.null(category)) {
        entries <- Filter(function(e) identical(e$category, category), entries)
      }
      if (!is.null(template_name)) {
        entries <- Filter(function(e) identical(e$template_name, template_name),
                          entries)
      }
      entries <- entries[order(vapply(entries, function(e)
        as.numeric(e$confidence %||% 0), numeric(1)), decreasing = TRUE)]
      head(entries, limit)
    }
  )
)

`%||%` <- function(a, b) if (is.null(a)) b else a


#' Knowledge retriever: format past decisions and context for prompt injection.
#'
#' @section Methods:
#' \describe{
#'   \item{\code{$retrieve(template_name=NULL, scope=NULL)}}{General retrieval.}
#'   \item{\code{$retrieve_for_analysis(template_name=NULL, scope=NULL)}}{
#'         Emphasise column mappings and data-quality patterns.}
#'   \item{\code{$retrieve_for_adapter(template_name=NULL, scope=NULL)}}{
#'         Emphasise past adapter code and troubleshooting notes.}
#' }
#'
#' @export
KnowledgeRetriever <- R6::R6Class(
  "KnowledgeRetriever",
  public = list(
    decision_store = NULL,
    context_store  = NULL,
    max_decisions  = 5L,
    max_context    = 5L,
    initialize = function(decision_store, context_store,
                          max_decisions = 5L, max_context = 5L) {
      self$decision_store <- decision_store
      self$context_store  <- context_store
      self$max_decisions  <- max_decisions
      self$max_context    <- max_context
    },
    retrieve = function(template_name = NULL, scope = NULL) {
      d <- self$decision_store$query(template_name = template_name,
                                     limit = self$max_decisions)
      c <- self$context_store$query(scope = scope, template_name = template_name,
                                    limit = self$max_context)
      private$.format_block("KNOWLEDGE BASE",
        list(`PAST DECISIONS:` = private$.format_decisions(d),
             `LEARNED PATTERNS:` = private$.format_context(c)))
    },
    retrieve_for_analysis = function(template_name = NULL, scope = NULL) {
      d <- self$decision_store$query(template_name = template_name,
                                     limit = self$max_decisions)
      c <- self$context_store$query(scope = scope, category = "column_mapping",
                                    template_name = template_name,
                                    limit = self$max_context)
      q <- self$context_store$query(scope = scope, category = "data_quality",
                                    template_name = template_name,
                                    limit = 3L)
      private$.format_block("ANALYSIS KNOWLEDGE",
        list(`PAST COLUMN MAPPINGS:` = private$.format_mappings(d),
             `MAPPING PATTERNS:`     = private$.format_context(c),
             `DATA QUALITY NOTES:`   = private$.format_context(q)))
    },
    retrieve_for_adapter = function(template_name = NULL, scope = NULL) {
      d <- self$decision_store$query(template_name = template_name,
                                     limit = self$max_decisions)
      d <- Filter(function(r) nzchar(as.character(r$adapter_code %||% "")), d)
      c <- self$context_store$query(scope = scope, category = "adapter_pattern",
                                    template_name = template_name,
                                    limit = self$max_context)
      t <- self$context_store$query(scope = scope, category = "troubleshooting",
                                    template_name = template_name,
                                    limit = 3L)
      lines <- character()
      if (length(d)) {
        for (r in d) {
          lines <- c(lines,
                     sprintf("  --- Run %s (%s) ---", r$run_id, r$template_name),
                     sprintf("  Config: %s",
                             jsonlite::toJSON(r$config_dict, auto_unbox = TRUE)),
                     if (nzchar(as.character(r$outcome_notes %||% "")))
                       sprintf("  Outcome: %s", r$outcome_notes),
                     "  Code:", r$adapter_code, "")
        }
      }
      private$.format_block("ADAPTER KNOWLEDGE",
        list(`PAST ADAPTER CODE:` = paste(lines, collapse = "\n"),
             `ADAPTER PATTERNS:`  = private$.format_context(c),
             `TROUBLESHOOTING NOTES:` = private$.format_context(t)))
    }
  ),
  private = list(
    .format_decisions = function(decisions) {
      if (!length(decisions)) return("")
      lines <- unlist(lapply(decisions, function(d) {
        c(sprintf("  Run %s (%s):", d$run_id, d$timestamp),
          sprintf("    Template: %s", d$template_name),
          sprintf("    Config: %s", jsonlite::toJSON(d$config_dict, auto_unbox = TRUE)),
          if (nzchar(as.character(d$outcome_notes %||% "")))
            sprintf("    Outcome: %s", d$outcome_notes),
          if (length(d$outcome_metrics))
            sprintf("    Metrics: %s",
                    jsonlite::toJSON(d$outcome_metrics, auto_unbox = TRUE)))
      }))
      paste(lines, collapse = "\n")
    },
    .format_mappings = function(decisions) {
      if (!length(decisions)) return("")
      lines <- unlist(lapply(decisions, function(d) {
        c(sprintf("  Run %s (%s):", d$run_id, d$template_name),
          if (length(d$auto_detected))
            sprintf("    Auto-detected: %s",
                    jsonlite::toJSON(d$auto_detected, auto_unbox = TRUE)),
          if (length(d$user_answers))
            sprintf("    User-provided: %s",
                    jsonlite::toJSON(d$user_answers, auto_unbox = TRUE)),
          if (nzchar(as.character(d$outcome_notes %||% "")))
            sprintf("    Outcome: %s", d$outcome_notes))
      }))
      paste(lines, collapse = "\n")
    },
    .format_context = function(entries) {
      if (!length(entries)) return("")
      lines <- unlist(lapply(entries, function(e) {
        c(sprintf("  [%s] %s (%.0f%%)",
                  e$category, e$title, 100 * as.numeric(e$confidence %||% 0)),
          sprintf("    %s", e$content))
      }))
      paste(lines, collapse = "\n")
    },
    .format_block = function(banner, sections) {
      sections <- sections[vapply(sections, function(s) nzchar(s), logical(1))]
      if (length(sections) == 0) return("")
      body <- mapply(function(name, body) paste(name, body, sep = "\n"),
                     names(sections), sections, USE.NAMES = FALSE)
      sprintf("--- %s ---\n%s\n--- END %s ---",
              banner, paste(body, collapse = "\n\n"), banner)
    }
  )
)


#' List built-in templates.
#' @return A named character vector mapping template name to description.
#' @export
list_templates <- function() {
  c(segmentation = "Entity segmentation using KMeans/hierarchical clustering",
    market_mix   = "Adstock + saturation + ridge regression for media ROI")
}
