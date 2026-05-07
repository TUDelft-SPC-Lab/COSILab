# libraries
library(readxl)
library(lavaan)

# read data
setwd("C:\\Users\\arthu\\OneDrive\\Documents\\Projects\\cosilab-pre") # change it to YOUR directory
raw <- read_excel(
  "INGroup - Prolific - Pre annotation survey_May 5, 2026_16.38.xlsx",
  sheet = 1,
  .name_repair = "minimal"
)
codebook <- read_excel("pre_annotation_survey_codebook.xlsx", sheet = 1)

# The first row contains survey question text, and row 3 onward contains individual responses
question_text <- as.character(raw[1, ])
survey_raw <- raw[-1, , drop = FALSE]

# The codebook is used as a position-based lookup table:
# codebook row i describes raw/survey_raw column i.
cat("raw columns:", ncol(raw), "| codebook rows:", nrow(codebook), "\n")
if (ncol(raw) != nrow(codebook)) {
  stop("Raw data columns and codebook rows do not have the same length.")
}

# a function that changes response to numeric value
response_to_numeric <- function(response, scale_min, scale_max) {
  response_text <- trimws(as.character(response))
  response_text[response_text == ""] <- NA

  values <- suppressWarnings(as.numeric(response_text))

  needs_mapping <- is.na(values) & !is.na(response_text)

  # Most responses include a score at the beginning, e.g. "1 (never)".
  leading_number <- regexpr("^[-+]?[0-9]+([.][0-9]+)?", response_text)
  has_leading_number <- needs_mapping & leading_number > 0
  leading_values <- rep(NA_character_, length(response_text))
  leading_values[!is.na(leading_number) & leading_number > 0] <- regmatches(response_text, leading_number)
  values[has_leading_number] <- as.numeric(leading_values[has_leading_number])

  needs_mapping <- is.na(values) & !is.na(response_text)

  # Schwartz-style responses put the score in parentheses, e.g. "Important (4)".
  parenthetical_number <- regexpr("[(][-+]?[0-9]+([.][0-9]+)?[)]", response_text)
  has_parenthetical_number <- needs_mapping & parenthetical_number > 0
  parenthetical_values <- rep(NA_character_, length(response_text))
  parenthetical_values[!is.na(parenthetical_number) & parenthetical_number > 0] <- regmatches(response_text, parenthetical_number)
  values[has_parenthetical_number] <- as.numeric(
    gsub("[()]", "", parenthetical_values[has_parenthetical_number])
  )

  needs_mapping <- is.na(values) & !is.na(response_text)

  if (any(needs_mapping) && scale_min == 1 && scale_max == 5) {
    lookup <- c(
      "Strongly disagree" = 1,
      "Somewhat disagree" = 2,
      "Neither agree nor disagree" = 3,
      "Somewhat agree" = 4,
      "Strongly agree" = 5
    )
    values[needs_mapping] <- unname(lookup[response_text[needs_mapping]])
  }

  needs_mapping <- is.na(values) & !is.na(response_text)

  if (any(needs_mapping) && scale_min == 1 && scale_max == 6) {
    lookup <- c(
      "Strongly disagree" = 1,
      "Disagree" = 2,
      "Slightly disagree" = 3,
      "Slightly agree" = 4,
      "Agree" = 5,
      "Strongly agree" = 6,
      "Storngly agree" = 6
    )
    values[needs_mapping] <- unname(lookup[response_text[needs_mapping]])
  }

  values
}

score_item <- function(response, scale_min, scale_max, reverse = FALSE) {
  values <- response_to_numeric(response, scale_min, scale_max)

  if (isTRUE(reverse)) {
    values <- scale_min + scale_max - values
  }

  values
}

codebook$col_index <- seq_len(nrow(codebook))
measure_codebook <- codebook[!is.na(codebook$measure), ]
measure_names <- unique(measure_codebook$measure)
item_scores <- survey_raw

conversion_checks <- list()

for (i in measure_codebook$col_index) {
  raw_i <- survey_raw[[i]]

  parsed_i <- response_to_numeric(
    raw_i,
    codebook$scale_min[[i]],
    codebook$scale_max[[i]]
  )

  scored_i <- parsed_i

  if (isTRUE(codebook$reverse[[i]])) {
    scored_i <- codebook$scale_min[[i]] + codebook$scale_max[[i]] - parsed_i
  }

  item_scores[[i]] <- scored_i

  check_df <- data.frame(
    col_index = i,
    label = codebook$label[[i]],
    measure = codebook$measure[[i]],
    reverse = codebook$reverse[[i]],
    raw_response = as.character(raw_i),
    parsed_numeric = parsed_i,
    final_score = scored_i,
    stringsAsFactors = FALSE
  )

  conversion_checks[[as.character(i)]] <- check_df

  # cat("\n==============================\n")
  # cat("Column:", i, "\n")
  # cat("Label:", codebook$label[[i]], "\n")
  # cat("Measure:", codebook$measure[[i]], "\n")
  # cat("Reverse:", codebook$reverse[[i]], "\n")
  # cat("==============================\n")

  #  print(
  #    unique(check_df[, c("raw_response", "parsed_numeric", "final_score")])
  #  )

  unmapped <- unique(
    check_df[
      !is.na(check_df$raw_response) &
        check_df$raw_response != "" &
        is.na(check_df$parsed_numeric),
      c("raw_response", "parsed_numeric", "final_score")
    ]
  )

  if (nrow(unmapped) > 0) {
    cat("\nWARNING: These responses were not mapped to numeric:\n")
    print(unmapped)
  }
}

conversion_check_all <- do.call(rbind, conversion_checks)

response_id_col <- which(codebook$label == "ResponseId")[1]
measure_scores <- data.frame(ResponseID = survey_raw[[response_id_col]])

# attention check
attn_correct_answers <- c(
  "Q7" = 4,
  "Q14" = 5,
  "Q29" = 1,
  "Q52" = 2
)

rfq_score_c <- function(values) {
  scored <- rep(0, length(values))
  scored[is.na(values)] <- NA_real_
  scored[values == 1] <- 3
  scored[values == 2] <- 2
  scored[values == 3] <- 1
  scored
}

rfq_score_u <- function(values) {
  scored <- rep(0, length(values))
  scored[is.na(values)] <- NA_real_
  scored[values == 5] <- 1
  scored[values == 6] <- 2
  scored[values == 7] <- 3
  scored
}

for (measure in measure_names) {
  item_cols <- measure_codebook$col_index[measure_codebook$measure == measure]

  if (measure == "attn") {
    item_labels <- as.character(codebook$label[item_cols])

    correct_answers <- sapply(item_cols, function(i) {
      label_i <- as.character(codebook$label[[i]])

      if (!(label_i %in% names(attn_correct_answers))) {
        stop(paste("No correct answer provided for attention check label:", label_i))
      }

      response_to_numeric(
        attn_correct_answers[[label_i]],
        codebook$scale_min[[i]],
        codebook$scale_max[[i]]
      )
    })

    correct_matrix <- sweep(
      as.data.frame(item_scores[item_cols]),
      MARGIN = 2,
      STATS = correct_answers,
      FUN = "=="
    )

    # NA / blank counts as incorrect
    correct_matrix[is.na(correct_matrix)] <- FALSE

    measure_scores[["attn_prop"]] <- rowMeans(correct_matrix)
    measure_scores[["attn_n_correct"]] <- rowSums(correct_matrix)
  } else if (measure == "AHS-12") {
    # AHS-12: Requires CFA with reverse-coded items and latent factor scores
    ahs_data <- as.data.frame(item_scores[item_cols])

    # Rename columns to V1, V2, ..., V12 for CFA model
    colnames(ahs_data) <- paste0("V", 1:12)

    # Apply reverse coding to items 7, 8, 9 (columns 7, 8, 9 within ahs_data)
    # Recoding: 8 - value
    ahs_data[, 7:9] <- 8 - ahs_data[, 7:9]

    # Remove rows with all NA values to avoid CFA errors
    complete_rows_idx <- which(rowSums(!is.na(ahs_data)) > 0)
    ahs_data_complete <- ahs_data[complete_rows_idx, ]

    if (length(complete_rows_idx) > 0) {
      # CFA model with 4 factors as specified in the paper
      ahs_model <- "
        Causality =~ V1 + V2 + V3
        Att_Contradiction =~ V4 + V5 + V6
        Percep_Change =~ V7 + V8 + V9
        Locus_Attention =~ V10 + V11 + V12
      "

      tryCatch(
        {
          ahs_fit <- cfa(ahs_model, ahs_data_complete, estimator = "MLR", std.ov = TRUE, std.lv = TRUE)
          ahs_factor_scores <- predict(ahs_fit)

          # Map factor scores back to original row indices
          # Initialize with full number of rows (all survey responses)
          factor_scores_full <- data.frame(
            Causality = rep(NA_real_, nrow(ahs_data)),
            Att_Contradiction = rep(NA_real_, nrow(ahs_data)),
            Percep_Change = rep(NA_real_, nrow(ahs_data)),
            Locus_Attention = rep(NA_real_, nrow(ahs_data))
          )
          factor_scores_full[complete_rows_idx, ] <- ahs_factor_scores

          # Add factor scores to measure_scores
          measure_scores[["AHS12_Causality"]] <- factor_scores_full$Causality
          measure_scores[["AHS12_Att_Contradiction"]] <- factor_scores_full$Att_Contradiction
          measure_scores[["AHS12_Percep_Change"]] <- factor_scores_full$Percep_Change
          measure_scores[["AHS12_Locus_Attention"]] <- factor_scores_full$Locus_Attention
        },
        error = function(e) {
          warning(paste("AHS-12 CFA failed:", e$message))
          # Fallback to mean scores if CFA fails
          measure_scores[["AHS-12"]] <<- rowMeans(ahs_data, na.rm = TRUE)
          measure_scores[["AHS-12"]][is.nan(measure_scores[["AHS-12"]])] <<- NA
        }
      )
    }
  } else if (measure == "RFQ") {
    item_cols <- sort(item_cols)
    rfq_raw <- lapply(item_cols, function(i) {
      response_to_numeric(
        survey_raw[[i]],
        codebook$scale_min[[i]],
        codebook$scale_max[[i]]
      )
    })
    rfq_raw <- as.data.frame(rfq_raw)

    measure_scores[["RFQ_c"]] <- rowMeans(
      data.frame(
        rfq_score_c(rfq_raw[[1]]),
        rfq_score_c(rfq_raw[[2]]),
        rfq_score_c(rfq_raw[[3]]),
        rfq_score_c(rfq_raw[[4]]),
        rfq_score_c(rfq_raw[[5]]),
        rfq_score_c(rfq_raw[[6]])
      ),
      na.rm = TRUE
    )
    measure_scores[["RFQ_c"]][is.nan(measure_scores[["RFQ_c"]])] <- NA

    measure_scores[["RFQ_u"]] <- rowMeans(
      data.frame(
        rfq_score_u(rfq_raw[[2]]),
        rfq_score_u(rfq_raw[[4]]),
        rfq_score_u(rfq_raw[[5]]),
        rfq_score_u(rfq_raw[[6]]),
        rfq_score_u(rfq_raw[[7]]),
        rfq_score_u(rfq_raw[[8]])
      ),
      na.rm = TRUE
    )
    measure_scores[["RFQ_u"]][is.nan(measure_scores[["RFQ_u"]])] <- NA
  } else {
    measure_values <- rowMeans(as.data.frame(item_scores[item_cols]), na.rm = TRUE)
    measure_values[is.nan(measure_values)] <- NA
    measure_scores[[measure]] <- measure_values
  }
}
# write.csv(measure_scores, 'person_measure_scores.csv', row.names = FALSE)

# Columns that are NOT part of any measure
non_measure_cols <- which(is.na(codebook$measure) | codebook$measure == "")

# Pull those columns from original survey data
non_measure_data <- survey_raw[non_measure_cols]

# Rename them using codebook labels
# make.unique() prevents duplicated labels from breaking the data frame
names(non_measure_data) <- make.unique(as.character(codebook$label[non_measure_cols]))

# Combine non-measure columns with measure mean scores
# measure_scores already contains ResponseID, so remove it if ResponseId is already in non_measure_data
final_data <- cbind(
  non_measure_data,
  measure_scores[, setdiff(names(measure_scores), "ResponseID"), drop = FALSE]
)

final_data$attn <- NULL
# Check
View(final_data)
head(final_data)

# Save
# write.csv(final_data, "final_data_with_measure_scores.csv", row.names = FALSE)

# process age
age_col <- grep("^age$", names(final_data), ignore.case = TRUE, value = TRUE)

age_num <- suppressWarnings(as.numeric(final_data[[age_col]]))

n_age_bins <- 4

age_min <- floor(min(age_num, na.rm = TRUE))
age_max <- ceiling(max(age_num, na.rm = TRUE))

bin_width <- ceiling((age_max - age_min + 1) / n_age_bins)

breaks <- seq(
  from = age_min,
  to = age_min + bin_width * n_age_bins,
  by = bin_width
)

labels <- paste0(
  breaks[-length(breaks)],
  "-",
  breaks[-1] - 1
)

final_data$age_quantized <- cut(
  age_num,
  breaks = breaks,
  labels = labels,
  include.lowest = TRUE,
  right = FALSE
)

final_data[[age_col]] <- NULL
table(final_data$age_quantized, useNA = "ifany")

# recode country to region
recode_country_to_region <- function(x) {
  x <- trimws(tolower(as.character(x)))
  x[x == ""] <- NA

  out <- rep(NA_character_, length(x))

  # Missing / invalid / unclear responses
  out[is.na(x) | x %in% c("s", "na", "n/a", "none")] <- "Missing / unclear"

  # Europe
  out[x %in% c(
    "the netherlands",
    "netherlands",
    "england",
    "united kingdom",
    "uk",
    "germany",
    "switzerland",
    "hungary",
    "macedonia",
    "north macedonia",
    "romania",
    "ireland",
    "italy",
    "scotland",
    "latvia",
    "wales",
    "scotland, united kingdom",
    "britain"
  )] <- "Europe"

  # Asia
  out[x %in% c(
    "india",
    "china",
    "japan",
    "philippines",
    "myanmar",
    "pakistan"
  )] <- "Asia"

  # North America
  out[x %in% c(
    "united states",
    "usa",
    "us",
    "u.s.",
    "u.s.a.",
    "canada",
    "united states of america",
    "united stations of america"
  )] <- "North America"

  # Small / rare regions
  out[x %in% c(
    "trinidad and tobago",
    "new zealand",
    "lesotho",
    "mexico",
    "nigeria"
  )] <- "Other region"

  unmapped <- unique(x[is.na(out) & !is.na(x)])
  unmapped <- unmapped[!unmapped %in% c("s", "na", "n/a", "none")]
  if (length(unmapped) > 0) {
    cat("Unmapped region responses becoming 'Other region':\n")
    print(unmapped)
  }

  # Anything not mapped
  out[is.na(out)] <- "Other region"

  out
}

final_data$birth_region <- recode_country_to_region(final_data$Q56)
final_data$residence_region <- recode_country_to_region(final_data$Q55)

# Drop original country
final_data$Q56 <- NULL
final_data$Q55 <- NULL

table(final_data$birth_region, useNA = "ifany")

# process nationality
recode_nationality_to_region <- function(x) {
  x <- trimws(tolower(as.character(x)))
  x[x == ""] <- NA

  out <- rep(NA_character_, length(x))

  # Missing / invalid / unclear responses
  out[is.na(x) | x %in% c("s", "na", "n/a", "none")] <- "Missing / unclear"

  # Europe
  out[x %in% c(
    "dutch",
    "british",
    "german",
    "germany",
    "hungary & uk",
    "hungarian",
    "macedonian",
    "irish",
    "new zealand/british",
    "wales",
    "welsh",
    "scottish",
    "uk",
    "united kingdom",
    "english",
    "swedish"
  )] <- "Europe"

  # Asia
  out[x %in% c(
    "indian",
    "chinese",
    "japanese",
    "burmese",
    "pakistani"
  )] <- "Asia"

  # North America
  out[x %in% c(
    "american",
    "usa",
    "canadian",
    "united states",
    "us"
  )] <- "North America"

  # Rare / small region
  out[x %in% c(
    "trinidadian",
    "mexican",
    "lesotho",
    "nigerian"
  )] <- "Other region"

  unmapped <- unique(x[is.na(out) & !is.na(x)])
  unmapped <- unmapped[!unmapped %in% c("s", "na", "n/a", "none")]
  if (length(unmapped) > 0) {
    cat("Unmapped nationality responses becoming 'Other region':\n")
    print(unmapped)
  }

  # Anything not mapped
  out[is.na(out)] <- "Other region"

  out
}

final_data$nationality_region <- recode_nationality_to_region(final_data$Q54)

# Drop original nationality
final_data$Q54 <- NULL

# Check distribution
table(final_data$nationality_region, useNA = "ifany")

# ethinicity
recode_ethnicity_to_group <- function(x) {
  x <- trimws(tolower(as.character(x)))
  x[x == ""] <- NA

  out <- rep(NA_character_, length(x))

  # Missing / unclear
  out[
    is.na(x) |
      x %in% c("na", "n/a", "none", "(i don't know - what are the options?)")
  ] <- "Missing / unclear"

  # White / European
  out[x %in% c(
    "dutch",
    "white",
    "german",
    "caucasian",
    "caucasion",
    "white european",
    "european",
    "hungarian",
    "macedonian",
    "white british",
    "white - british",
    "british",
    "white non hispanic",
    "white scottish"
  )] <- "White / European"

  # Asian
  out[x %in% c(
    "south asian",
    "indian",
    "han",
    "asian",
    "japan",
    "japanese",
    "chinese",
    "burmese"
  )] <- "Asian"

  # Small / potentially identifiable groups
  out[x %in% c(
    "african decent",
    "african descent",
    "white/hispanic",
    "english indian",
    "african",
    "latin",
    "hispanic",
    "american"
  )] <- "Other"

  unmapped <- unique(x[is.na(out) & !is.na(x)])
  unmapped <- unmapped[!unmapped %in% c("na", "n/a", "none", "(i don't know - what are the options?)")]
  if (length(unmapped) > 0) {
    cat("Unmapped ethnicity responses becoming 'Other':\n")
    print(unmapped)
  }

  # Anything else not mapped
  out[is.na(out)] <- "Other"

  out
}

final_data$ethnicity_group <- recode_ethnicity_to_group(final_data$Q57)

# Drop original ethnicity free response / detailed answer
final_data$Q57 <- NULL

# Drop Qualtrics metadata columns, but keep id1-id4
metadata_cols_to_drop <- c(
  "StartDate",
  "EndDate",
  "Status",
  "IPAddress",
  "Progress",
  "Duration (in seconds)",
  "Finished",
  "RecordedDate",
  "ResponseId",
  "RecipientLastName",
  "RecipientFirstName",
  "RecipientEmail",
  "ExternalReference",
  "LocationLatitude",
  "LocationLongitude",
  "DistributionChannel",
  "UserLanguage"
)

# Check which columns are actually present
metadata_cols_to_drop <- metadata_cols_to_drop[
  metadata_cols_to_drop %in% names(final_data)
]

# Drop them
final_data <- final_data[, !(names(final_data) %in% metadata_cols_to_drop)]

# Check
table(final_data$ethnicity_group, useNA = "ifany")

# Remove rows with NA in gender column
initial_rows <- nrow(final_data)
final_data <- final_data[!is.na(final_data$`gender`), ]
removed_rows <- initial_rows - nrow(final_data)
cat("Rows removed due to missing gender:", removed_rows, "\n")
cat("Remaining rows:", nrow(final_data), "\n")

write.csv(final_data, "pre_annotation_survey_processed.csv", row.names = FALSE)
