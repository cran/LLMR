## -----------------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE, comment = "#>",
  eval = identical(tolower(Sys.getenv("LLMR_RUN_VIGNETTES", "false")), "true") )

## -----------------------------------------------------------------------------
# library(LLMR)
# library(dplyr)
# cfg_ds     <- llm_config("deepseek", "deepseek-chat")
# cfg_groq1  <- llm_config("groq",     "llama-3.1-8b-instant")
# cfg_groq   <- llm_config("groq",     "openai/gpt-oss-20b")

## -----------------------------------------------------------------------------
# experiments <- build_factorial_experiments(
#   configs       = list(cfg_ds, cfg_groq1, cfg_groq),
#   user_prompts  = c("Summarize in one sentence: The Apollo program.",
#                     "List two benefits of green tea."),
#   system_prompts = c("Be concise.")
# )
# experiments

## -----------------------------------------------------------------------------
# setup_llm_parallel(workers = 10)
# res_unstructured <- call_llm_par(experiments, progress = TRUE)
# reset_llm_parallel()
# res_unstructured |>
#   select(provider, model, user_prompt_label, response_text, finish_reason) |>
#   head()

## -----------------------------------------------------------------------------
# schema <- list(
#   type = "object",
#   properties = list(
#     answer = list(type="string"),
#     keywords = list(type="array", items = list(type="string"))
#   ),
#   required = list("answer","keywords"),
#   additionalProperties = FALSE
# )
# 
# experiments2 <- experiments
# experiments2$config <- lapply(experiments2$config, enable_structured_output, schema = schema)
# 
# setup_llm_parallel(workers = 10)
# res_structured <- call_llm_par_structured(experiments2 , .fields = c("answer","keywords") )
# reset_llm_parallel()
# 
# res_structured |>
#   select(provider, model, user_prompt_label, structured_ok, answer) |>
#   head()
# 
# 

