## -----------------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE, comment = "#>",
  eval = identical(tolower(Sys.getenv("LLMR_RUN_VIGNETTES", "false")), "true")
)

## ----setup--------------------------------------------------------------------
# library(LLMR)
# cfg <- llm_config("groq", "openai/gpt-oss-20b", temperature = 0, seed = 110)

## ----audit--------------------------------------------------------------------
# log_path <- tempfile(fileext = ".jsonl")
# llm_log_enable(log_path)
# 
# r <- call_llm(cfg, "In one word, the capital of Senegal?")
# 
# llm_log_disable()
# jsonlite::stream_in(file(log_path), verbose = FALSE)[
#   , c("provider", "model", "model_version", "finish_reason", "status")]

## ----replicate----------------------------------------------------------------
# reviews <- tibble::tibble(text = c(
#   "The course changed how I think.",
#   "Lectures were fine, assignments tedious.",
#   "A complete waste of an afternoon."
# ))
# 
# cfg_warm <- llm_config("groq", "openai/gpt-oss-20b", temperature = 1)
# 
# reps <- llm_replicate(
#   reviews, sentiment,
#   prompt  = "Sentiment of '{text}'. Answer with exactly one word: positive, negative, or neutral.",
#   .config = cfg_warm, .times = 5
# )
# 
# ag <- llm_agreement(reps, prefix = "sentiment")
# ag
# ag$by_row

## ----methods------------------------------------------------------------------
# res <- call_llm_par(
#   build_factorial_experiments(
#     configs = cfg,
#     user_prompts = c("Classify: 'great work'", "Classify: 'do better'")
#   )
# )
# cat(llm_methods_text(res, task = "to classify short feedback messages"))

## ----usage--------------------------------------------------------------------
# llm_usage(res)
# 
# my_prices <- data.frame(
#   model  = "openai/gpt-oss-20b",
#   input  = 0.10,   # $ per million input tokens; check your provider's page
#   output = 0.50,
#   cached = 0.05
# )
# llm_usage(res, price_table = my_prices)$cost_estimate

## ----batch, eval = FALSE------------------------------------------------------
# job <- llm_batch_submit(
#   cfg,
#   c("Classify: 'superb'", "Classify: 'awful'", "Classify: 'fine, I guess'"),
#   state_path = "sentiment_batch.rds"
# )
# llm_batch_status(job)
# 
# # hours later, in a fresh session:
# res_batch <- llm_batch_fetch("sentiment_batch.rds")
# llm_usage(res_batch)

