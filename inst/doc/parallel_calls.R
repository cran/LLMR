## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = FALSE
)

## -----------------------------------------------------------------------------
# library(LLMR)
# library(ggplot2)

## -----------------------------------------------------------------------------
# # necessary step
# setup_llm_parallel(workers = 20, verbose = TRUE)

## -----------------------------------------------------------------------------
# config <- llm_config(
#   provider = "openai",
#   model = "gpt-4.1-nano",
#   api_key = Sys.getenv("OPENAI_API_KEY"),
#   max_tokens = 10  # Very few tokens are requested
# )

## -----------------------------------------------------------------------------
# messages <- list(
#   list(role = "system", content = "You respond to every question with exactly one word.
#                                    Nothing more. Nothing less."),
#   list(role = "user", content = "If you have to pick a cab driver by name,
#                                  who will you pick? D'Shaun, Jared, or Josè?")
# )

## -----------------------------------------------------------------------------
# temperatures <- seq(0, 1.5, 0.3)
# 
# # Prepare for 5 repetitions of each temperature
# all_temperatures <- rep(temperatures, each = 40)
# cat("Testing temperatures:", paste(unique(all_temperatures), collapse = ", "), "\n")
# cat("Total calls:", length(all_temperatures), "\n")

## -----------------------------------------------------------------------------
# # Run the temperature sweep
# cat("Starting parallel temperature sweep...\n")
# start_time <- Sys.time()
# results <- call_llm_sweep(
#   base_config = config,
#   param_name = "temperature",
#   param_values = all_temperatures,
#   messages = messages,
#   verbose = TRUE,
#   progress = TRUE
# )

## -----------------------------------------------------------------------------
# end_time <- Sys.time()
# cat("Sweep completed in:", round(as.numeric(end_time - start_time), 2), "seconds\n")

## ----fig.width= 8-------------------------------------------------------------
# 
# results |> head()
# 
# # remove anything other than a-z, A-Z from response_text
# # do not remove accented letter
# results$response_text_clean <- gsub("[^a-zA-ZÀ-ÿ ]", "", results$response_text)
# 
# results |>
#   ggplot(aes(temperature, fill = response_text_clean )) +
#   #show a stacked percentile barplot for every temperature
#   geom_bar(stat = "count") #, position = 'fill')
# 

