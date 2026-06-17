## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE, comment = "#>",
  eval = identical(tolower(Sys.getenv("LLMR_RUN_VIGNETTES", "false")), "true")
)

## ----setup, message=FALSE, warning=FALSE--------------------------------------
# library(LLMR)
# library(dplyr)
# 
# # Configure an open model endpoint
# cfg <- llm_config(
#   provider = "groq",
#   model    = "llama-3.1-8b-instant"
# )

## ----stimuli------------------------------------------------------------------
# dilemmas <- tibble::tibble(
#   condition = c("Switch", "Footbridge"),
#   scenario = c(
#     "A runaway trolley is heading down the tracks toward five workers who will be killed. You are standing next to a switch. If you pull the switch, the trolley will be diverted onto a side track where it will kill one worker. Do you pull the switch?",
#     "A runaway trolley is heading toward five workers. You are standing on a footbridge above the tracks next to a large stranger. If you push the stranger onto the tracks below, his mass will stop the trolley, saving the five workers but killing the stranger. Do you push the stranger?"
#   )
# )

## ----mutate-------------------------------------------------------------------
# experiment_results <- dilemmas |>
#   llm_mutate(
#     response = c(
#       system = "You are a participant in a moral psychology experiment. Read the scenario and provide a definitive YES or NO decision, followed by a brief rationale. Enclose your decision in <decision>...</decision> tags and your reasoning in <rationale>...</rationale> tags.",
#       user = "{scenario}"
#     ),
#     .config = cfg,
#     .tags = c("decision", "rationale")
#   )

## ----inspect------------------------------------------------------------------
# experiment_results |>
#   select(condition, decision, rationale) |>
#   print(n = Inf)

