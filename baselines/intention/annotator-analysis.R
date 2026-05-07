
# set working directory 
# read in csv files 
annotation.final <- read.csv("df_final_LLM_parse.csv")
nrow(annotation.final)

survey <- read.csv("pre_annotation_survey_processed_gender_career_recoded.csv")

library(jsonlite)
library(dplyr)

# Robustly extract prolific_pid from data_json
extract_prolific_pid <- function(x) {
  
  if (is.na(x) || is.null(x)) {
    return(NA_character_)
  }
  
  # If data_json is stored as JSON string
  if (is.character(x)) {
    obj <- tryCatch(
      fromJSON(x, simplifyVector = FALSE),
      error = function(e) NULL
    )
  } else {
    # If data_json is already a list/object
    obj <- x
  }
  
  if (is.null(obj)) {
    return(NA_character_)
  }
  
  # recursive search for prolific_pid anywhere inside the JSON object
  find_key <- function(z, key = "prolific_pid") {
    
    if (is.list(z)) {
      if (!is.null(names(z)) && key %in% names(z)) {
        return(as.character(z[[key]][1]))
      }
      
      for (item in z) {
        result <- find_key(item, key)
        if (!is.na(result)) {
          return(result)
        }
      }
    }
    
    return(NA_character_)
  }
  
  find_key(obj)
}

# Extract PID from annotation.final$data_json
annotation.final$prolific_pid <- vapply(
  annotation.final$data_json,
  extract_prolific_pid,
  character(1)
)

# Check extraction
table(is.na(annotation.final$prolific_pid))
head(unique(annotation.final$prolific_pid), 20)

# If survey PID column is called PROLIFIC_PID, standardize its name
if ("PROLIFIC_PID" %in% names(survey)) {
  names(survey)[names(survey) == "PROLIFIC_PID"] <- "prolific_pid"
}

# Clean PID format on both sides
annotation.final$prolific_pid <- trimws(tolower(as.character(annotation.final$prolific_pid)))
survey$prolific_pid <- trimws(tolower(as.character(survey$prolific_pid)))

analysis_df <- annotation.final %>%
  left_join(survey, by = "prolific_pid")

cat("annotation rows:", nrow(annotation.final), "\n")
cat("merged rows:", nrow(analysis_df), "\n")

cat("unique annotation PIDs:", length(unique(annotation.final$prolific_pid)), "\n")
cat("unique survey PIDs:", length(unique(survey$prolific_pid)), "\n")
cat("matched annotation PIDs:", length(intersect(
  unique(annotation.final$prolific_pid),
  unique(survey$prolific_pid)
)), "\n")

# Keep only those who did both survey and annotation 
analysis_df_matched_only <- annotation.final %>%
  inner_join(survey, by = "prolific_pid")

names(analysis_df_matched_only)

# clean true false mention cols 
library(dplyr)
library(lme4)

df <- analysis_df_matched_only

mention_cols <- c(
  "mentions_emotion",
  "mentions_cues",
  "mentions_belief",
  "mentions_intention",
  "mentions_situation_characteristic"
)

df[mention_cols] <- lapply(df[mention_cols], function(x) {
  if (is.logical(x)) return(x)
  tolower(as.character(x)) %in% c("true", "t", "1", "yes")
})

df$prolific_pid <- as.factor(df$prolific_pid)
df$task_id <- as.factor(df$task_id)
df$gender <- as.factor(df$gender)
df$career_stage <- as.factor(df$career_stage)

# mention appearance 
# total
mention_summary <- data.frame(
  mention = mention_cols,
  n_true = sapply(df[mention_cols], function(x) sum(x == TRUE, na.rm = TRUE)),
  n_total = sapply(df[mention_cols], function(x) sum(!is.na(x))),
  rate = sapply(df[mention_cols], function(x) mean(x == TRUE, na.rm = TRUE))
)

print(mention_summary)

# task-level 
task_mention_summary <- df %>%
  group_by(task_id) %>%
  summarise(
    emotion_rate = mean(mentions_emotion == TRUE, na.rm = TRUE),
    cues_rate = mean(mentions_cues == TRUE, na.rm = TRUE),
    belief_rate = mean(mentions_belief == TRUE, na.rm = TRUE),
    intention_rate = mean(mentions_intention == TRUE, na.rm = TRUE),
    situation_rate = mean(mentions_situation_characteristic == TRUE, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

head(task_mention_summary)

# H1: reflective functioning predicts belief/intention/emotion mention 
df <- df %>%
  mutate(
    RFQ_c_z = as.numeric(scale(RFQ_c)),
    RFQ_u_z = as.numeric(scale(RFQ_u)),
    csis_cs_z = as.numeric(scale(csis_cs))
  )


m_belief_rfq <- glmer(
  mentions_belief ~ RFQ_c_z + RFQ_u_z + gender +
    (1 | prolific_pid) + (1 | task_id),
  data = df,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)

summary(m_belief_rfq)
exp(fixef(m_belief_rfq))

# other mentions 
library(tidyr)
library(dplyr)
library(lme4)

df$row_id <- seq_len(nrow(df))

df_long <- df %>%
  select(
    row_id,
    prolific_pid,
    task_id,
    gender,
    RFQ_c_z,
    RFQ_u_z,
    csis_cs_z,
    mentions_emotion,
    mentions_cues,
    mentions_belief,
    mentions_intention
  ) %>%
  pivot_longer(
    cols = c(
      mentions_emotion,
      mentions_cues,
      mentions_belief,
      mentions_intention
    ),
    names_to = "mention_type",
    values_to = "mentioned"
  )

df_long$mentioned <- as.logical(df_long$mentioned)
df_long$mention_type <- as.factor(df_long$mention_type)

m_mentions_long <- glmer(
  mentioned ~ mention_type +
    RFQ_c_z * mention_type +
    RFQ_u_z * mention_type +
    csis_cs_z * mention_type +
    gender * mention_type +
    (1 | prolific_pid) +
    (1 | task_id) +
    (1 | row_id),
  data = df_long,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)

summary(m_mentions_long)

