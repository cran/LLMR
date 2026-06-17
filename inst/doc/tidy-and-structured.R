## -----------------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE, comment = "#>",
  eval = identical(tolower(Sys.getenv("LLMR_RUN_VIGNETTES", "false")), "true")
)

## -----------------------------------------------------------------------------
# library(LLMR)
# library(dplyr)
# 
# cfg_ds    <- llm_config("deepseek", "deepseek-chat")
# cfg_groq1 <- llm_config("groq",     "llama-3.1-8b-instant")
# cfg_groq  <- llm_config("groq",     "openai/gpt-oss-20b")

## -----------------------------------------------------------------------------
# words <- c("excellent", "awful", "fine")
# out <- llm_fn(
#   words,
#   prompt  = "Classify '{x}' as Positive, Negative, or Neutral.",
#   .config = cfg_ds,
#   .return = "columns"
# )
# out

## -----------------------------------------------------------------------------
# out_groq <- llm_fn(
#   words,
#   prompt  = "Classify '{x}' as Positive, Negative, or Neutral.",
#   .config = cfg_groq1,
#   .return = "columns"
# )
# out_groq

## -----------------------------------------------------------------------------
# schema <- list(
#   type = "object",
#   properties = list(
#     label = list(type = "string", description = "Sentiment label"),
#     score = list(type = "number", description = "Confidence 0..1")
#   ),
#   required = list("label", "score"),
#   additionalProperties = FALSE
# )
# 
# out_s <- llm_fn_structured(
#   x = words,
#   prompt  = "Classify '{x}' as Positive, Negative, or Neutral with confidence.",
#   .config = cfg_ds,
#   .schema = schema,
#   .fields = c("label", "score")
# )
# out_s

## -----------------------------------------------------------------------------
# df <- tibble::tibble(
#   id   = 1:3,
#   text = c("Cats are great pets", "The weather is bad", "I like tea")
# )
# 
# df_u <- df |>
#   llm_mutate(
#     answer  = "Give a short category for: {text}",
#     .config = cfg_groq,
#     .return = "columns"
#   )
# 
# df_u

## -----------------------------------------------------------------------------
# df |>
#   llm_mutate(
#     category = "Give a short category for: {text}",
#     .config = cfg_groq
#   )
# # Equivalent to: llm_mutate(category, prompt = "Give...", .config = cfg_groq)

## -----------------------------------------------------------------------------
# df |>
#   llm_mutate(
#     classified = c(
#       system = "You are a text classifier. One word only.",
#       user = "Category for: {text}"
#     ),
#     .config = cfg_ds
#   )

## -----------------------------------------------------------------------------
# schema <- list(
#   type = "object",
#   properties = list(
#     category = list(type = "string"),
#     confidence = list(type = "number")
#   ),
#   required = list("category", "confidence")
# )
# 
# # Using .structured = TRUE (equivalent to calling llm_mutate_structured)
# df |>
#   llm_mutate(
#     structured_result = "{text}",
#     .config = cfg_ds,
#     .structured = TRUE,
#     .schema = schema
#   )

## -----------------------------------------------------------------------------
# cities <- tibble::tibble(city = c("Cairo", "Lima", "Seoul"))
# 
# cities |>
#   llm_mutate(
#     geo = "Where is {city}? Give country and continent in their own tags.",
#     .config = cfg_groq1,
#     .system_prompt = paste(
#       "Use XML tags to specify different parts of the answer, but do not nest tags.",
#       "Return <country>...</country> and <continent>...</continent>."
#     ),
#     .tags = c("country", "continent")
#   )

## -----------------------------------------------------------------------------
# cities |>
#   llm_mutate(
#     geo = "Where is {city}? Give country and continent in their own tags.",
#     .config = cfg_groq1,
#     .tags = c("country", "continent"),
#     .rows_per_prompt = 3
#   )

## -----------------------------------------------------------------------------
# df <- data.frame(text = c("good", "bad", "fine"), stringsAsFactors = FALSE)
# LLMR::llm_preview(df, prompt = "Sentiment of: {text}", .rows_per_prompt = 2)

## ----eval=FALSE---------------------------------------------------------------
# out <- df |>
#   llm_mutate(sentiment = "One-word sentiment for: {text}", .config = cfg_groq)
# 
# llm_usage(out)       # counts + sent/received/total/reasoning tokens
# llm_failures(out)    # which rows failed or were truncated, and why

## -----------------------------------------------------------------------------
# schema2 <- list(
#   type = "object",
#   properties = list(
#     category  = list(type = "string"),
#     rationale = list(type = "string")
#   ),
#   required = list("category", "rationale"),
#   additionalProperties = FALSE
# )
# 
# # Traditional call
# df_s <- df |>
#   llm_mutate_structured(
#     annot,
#     prompt  = "Extract category and a one-sentence rationale for: {text}",
#     .config = cfg_groq,
#     .schema = schema2
#     # Because a schema is present, fields auto-hoist; you can also pass:
#     # .fields = c("category", "rationale")
#   )
# 
# df_s
# 
# # Or use shorthand
# df |>
#   llm_mutate_structured(
#     annot = "Extract category and rationale for: {text}",
#     .config = cfg_groq,
#     .schema = schema2
#   )

