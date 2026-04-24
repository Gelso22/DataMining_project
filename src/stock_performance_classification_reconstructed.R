# Stock Performance Classification - reconstructed analysis script
# -------------------------------------------------------------
# This script is a cleaned reconstruction of the workflow described in the
# final report. It is intended for a GitHub portfolio repository.
#
# Expected data source:
#   Kaggle: 200 Financial Indicators of US Stocks
#   Example file: data/2018_Financial_Data.csv
#
# The original local code was split across several exploratory files. This
# version keeps the core report logic: data cleaning, ratio-variable selection,
# missing-value handling, class-imbalance handling, logistic regression vs KNN,
# and final test-set assessment.

# -----------------------------
# 0. Packages
# -----------------------------

library(tidyverse)
library(tidymodels)
library(themis)       # step_upsample()
library(workflowsets) # workflow_set(), rank_results()

tidymodels_prefer()
set.seed(123)

# -----------------------------
# 1. User settings
# -----------------------------

DATA_PATH <- "2018_Financial_Data.csv"

# In the original report, after selecting ratio/multiple variables and removing
# redundant variables, the modelling dataset had roughly 80 predictors.
# If the automatic keyword selection below is not close to the original dataset,
# manually edit ratio_keywords or provide an explicit vector of variable names.
ratio_keywords <- c(
  "ratio", "margin", "turnover", "coverage", "return", "yield",
  "multiple", "roe", "roa", "roic", "per.share", "pershare",
  "debt.to", "debtto", "cash.flow", "cashflow", "free.cash",
  "freecash", "price.to", "priceto", "enterprise.value", "enterprisevalue",
  "ev.to", "evto", "payout", "graham"
)

# Extreme missingness thresholds used to mirror the report logic.
max_na_col_pct <- 50
max_na_row_pct <- 25

# -----------------------------
# 2. Helper functions
# -----------------------------

pct_missing <- function(x) mean(is.na(x)) * 100
pct_zero <- function(x) {
  if (!is.numeric(x)) return(NA_real_)
  mean(x == 0, na.rm = TRUE) * 100
}

normalise_var_name <- function(x) {
  x %>%
    str_to_lower() %>%
    str_replace_all("[^a-z0-9]", "")
}

summarise_missing_zero_by_col <- function(df) {
  tibble(
    variable = names(df),
    pct_na = map_dbl(df, pct_missing),
    pct_zero = map_dbl(df, pct_zero)
  ) %>%
    arrange(desc(pct_na))
}

summarise_missing_zero_by_row <- function(df) {
  tibble(
    row_id = rownames(df),
    pct_na = rowMeans(is.na(df)) * 100,
    pct_zero = apply(df, 1, function(z) mean(z == 0, na.rm = TRUE) * 100)
  ) %>%
    arrange(desc(pct_na))
}

# -----------------------------
# 3. Load and prepare raw data
# -----------------------------

raw_data <- read.csv(DATA_PATH, check.names = TRUE)

# The Kaggle file usually has company name in the first column, Sector near the
# end, a numeric next-year return column, and a binary Class column.
names(raw_data)[1] <- "Company"
if (!"Class" %in% names(raw_data)) {
  stop("The dataset must contain a binary target column named 'Class'.")
}
if (!"Sector" %in% names(raw_data)) {
  stop("The dataset must contain a categorical column named 'Sector'.")
}

# In the original files, the penultimate column was renamed Price_var.
# This column is useful for EDA but is not used as a predictor in classification.
if (!"Price_var" %in% names(raw_data)) {
  names(raw_data)[ncol(raw_data) - 1] <- "Price_var"
}

raw_data <- raw_data %>%
  mutate(
    Company = as.character(Company),
    Sector = as.factor(Sector),
    Class = factor(Class, levels = c(0, 1))
  ) %>%
  column_to_rownames("Company")

# Drop one problematic duplicated variable used in the old exploratory scripts,
# if present.
raw_data <- raw_data %>%
  select(-any_of("operatingProfitMargin"))

# -----------------------------
# 4. Exploratory summaries
# -----------------------------

numeric_feature_cols <- raw_data %>%
  select(where(is.numeric)) %>%
  select(-any_of("Price_var")) %>%
  names()

eda_features <- raw_data %>% select(all_of(numeric_feature_cols))

col_quality_raw <- summarise_missing_zero_by_col(eda_features)
row_quality_raw <- summarise_missing_zero_by_row(eda_features)

cat("Raw missing percentage:", round(mean(is.na(eda_features)) * 100, 2), "%\n")
cat("Raw zero percentage:", round(mean(as.matrix(eda_features) == 0, na.rm = TRUE) * 100, 2), "%\n")
cat("Raw class distribution:\n")
print(prop.table(table(raw_data$Class)))

# Optional plots used in the report-style EDA.
# Uncomment if you want to regenerate figures.
# dir.create("figures", showWarnings = FALSE)
# ggplot(col_quality_raw, aes(x = reorder(variable, pct_na), y = pct_na)) +
#   geom_col() + coord_flip() + labs(title = "Missing values by variable")
# ggsave("figures/missing_by_variable.png", width = 8, height = 10)
#
# ggplot(row_quality_raw, aes(x = seq_along(pct_na), y = pct_na)) +
#   geom_col() + labs(x = "Company", y = "% missing", title = "Missing values by company")
# ggsave("figures/missing_by_company.png", width = 8, height = 5)

# -----------------------------
# 5. Select ratio / relative financial variables
# -----------------------------

candidate_predictors <- setdiff(
  names(raw_data),
  c("Sector", "Price_var", "Class")
)

ratio_pattern <- paste(ratio_keywords, collapse = "|")
ratio_cols <- candidate_predictors[
  str_detect(str_to_lower(candidate_predictors), ratio_pattern)
]

# Fallback: if too few variables are detected by name, use a scale-based filter
# inspired by the old exploratory code. This is not perfect, but helps recover
# ratio-like variables in files with inconsistent naming.
if (length(ratio_cols) < 50) {
  medians <- raw_data %>%
    select(all_of(candidate_predictors)) %>%
    summarise(across(everything(), ~ median(.x, na.rm = TRUE))) %>%
    pivot_longer(everything(), names_to = "variable", values_to = "median")

  scale_based_cols <- medians %>%
    filter(between(median, -1000, 160)) %>%
    pull(variable)

  ratio_cols <- union(ratio_cols, scale_based_cols)
}

cat("Selected ratio-like predictors:", length(ratio_cols), "\n")

model_data <- raw_data %>%
  select(all_of(ratio_cols), Sector, Price_var, Class)

# -----------------------------
# 6. Remove clearly redundant variables
# -----------------------------

# Remove duplicated names after normalisation, e.g. PriceSalesRatio vs
# Price.to.Sales.Ratio. This mirrors the report's removal of variables that
# represent the same financial metric.
predictor_names <- setdiff(names(model_data), c("Sector", "Price_var", "Class"))
name_key <- normalise_var_name(predictor_names)
redundant_by_name <- predictor_names[duplicated(name_key)]

model_data <- model_data %>% select(-all_of(redundant_by_name))

# Remove exactly duplicated numeric columns, if any.
numeric_predictors <- model_data %>%
  select(where(is.numeric)) %>%
  select(-any_of("Price_var"))

if (ncol(numeric_predictors) > 1) {
  duplicated_numeric <- duplicated(as.list(numeric_predictors))
  model_data <- model_data %>%
    select(-all_of(names(numeric_predictors)[duplicated_numeric]))
}

cat("Predictors after redundancy removal:",
    ncol(model_data) - 3, "\n")

# -----------------------------
# 7. Remove extreme missingness
# -----------------------------

predictor_only <- model_data %>% select(-Price_var, -Class)

# Column filtering only on numeric predictors. Sector is retained.
col_na <- predictor_only %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), pct_missing)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "pct_na")

cols_to_remove <- col_na %>%
  filter(pct_na > max_na_col_pct) %>%
  pull(variable)

model_data <- model_data %>% select(-all_of(cols_to_remove))

# Row filtering based on predictor missingness.
row_na <- model_data %>%
  select(-Price_var, -Class) %>%
  select(where(is.numeric)) %>%
  is.na() %>%
  rowMeans() * 100

model_data <- model_data[row_na <= max_na_row_pct, ]
model_data <- model_data %>% drop_na(Class)

cat("Rows after missingness filtering:", nrow(model_data), "\n")
cat("Columns after missingness filtering:", ncol(model_data), "\n")
cat("Remaining missing percentage:",
    round(mean(is.na(model_data %>% select(-Price_var, -Class))) * 100, 2), "%\n")
cat("Class distribution after cleaning:\n")
print(prop.table(table(model_data$Class)))

# -----------------------------
# 8. Train/test split
# -----------------------------

# Price_var is not used in the predictive model; it is only useful for EDA.
classification_data <- model_data %>% select(-Price_var)

set.seed(123)
data_split <- initial_split(classification_data, prop = 0.80, strata = Class)
train_data <- training(data_split)
test_data  <- testing(data_split)

set.seed(123)
folds <- vfold_cv(train_data, v = 5, strata = Class)

# -----------------------------
# 9. Recipes
# -----------------------------

# Recipe 1 in the report:
# median imputation, zero-variance removal, dummy encoding for Sector, scaling.
recipe_1 <- recipe(Class ~ ., data = train_data) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Recipe 2 in the report:
# recipe 1 + correlation filter + upsampling of the minority class.
recipe_2 <- recipe(Class ~ ., data = train_data) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = tune()) %>%
  step_upsample(Class, over_ratio = tune())

# -----------------------------
# 10. Model specifications
# -----------------------------

logistic_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

knn_spec <- nearest_neighbor(
  neighbors = tune(),
  weight_func = "rectangular",
  dist_power = 2
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

metrics_cls <- metric_set(accuracy, roc_auc, sensitivity, specificity)

# -----------------------------
# 11. Workflow comparison
# -----------------------------

wf_set <- workflow_set(
  preproc = list(recipe_1 = recipe_1, recipe_2 = recipe_2),
  models = list(logistic_reg = logistic_spec, nearest_neighbor = knn_spec),
  cross = TRUE
)

grid_ctrl <- control_grid(save_pred = TRUE, save_workflow = TRUE)

set.seed(123)
wf_results <- wf_set %>%
  workflow_map(
    fn = "tune_grid",
    resamples = folds,
    grid = 10,
    metrics = metrics_cls,
    control = grid_ctrl,
    verbose = TRUE
  )

# Ranking similar to the workflow-rank plots in the report.
rank_accuracy <- rank_results(wf_results, rank_metric = "accuracy", select_best = TRUE)
rank_sensitivity <- rank_results(wf_results, rank_metric = "sensitivity", select_best = TRUE)
rank_roc_auc <- rank_results(wf_results, rank_metric = "roc_auc", select_best = TRUE)

print(rank_accuracy)
print(rank_sensitivity)
print(rank_roc_auc)

# -----------------------------
# 12. Final model assessment
# -----------------------------

# The report selected logistic regression with the second recipe because it gave
# a better trade-off for minority-class recognition, while remaining close in
# ROC AUC to the best base logistic model.
final_workflow_id <- "recipe_2_logistic_reg"

best_params <- wf_results %>%
  extract_workflow_set_result(final_workflow_id) %>%
  select_best(metric = "roc_auc")

final_workflow <- wf_results %>%
  extract_workflow(final_workflow_id) %>%
  finalize_workflow(best_params)

set.seed(123)
final_fit <- last_fit(
  final_workflow,
  split = data_split,
  metrics = metrics_cls
)

final_metrics <- collect_metrics(final_fit)
final_predictions <- collect_predictions(final_fit)

print(final_metrics)

# Confusion matrix on the test set.
conf_mat(final_predictions, truth = Class, estimate = .pred_class)

# Save reproducible outputs.
dir.create("outputs", showWarnings = FALSE)
saveRDS(wf_results, "outputs/workflow_cv_results.rds")
saveRDS(final_fit, "outputs/final_logistic_upsampled_fit.rds")
write.csv(final_metrics, "outputs/final_test_metrics.csv", row.names = FALSE)

# Optional final metric plot.
final_metrics %>%
  ggplot(aes(x = .metric, y = .estimate)) +
  geom_col() +
  ylim(0, 1) +
  labs(
    title = "Final model assessment on test set",
    x = NULL,
    y = "Estimate"
  )
# ggsave("outputs/final_test_metrics.png", width = 7, height = 4)
