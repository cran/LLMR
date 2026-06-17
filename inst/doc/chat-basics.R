## -----------------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE, comment = "#>",
  eval = identical(tolower(Sys.getenv("LLMR_RUN_VIGNETTES", "false")), "true")
)

## -----------------------------------------------------------------------------
# library(LLMR)
# 
# cfg_ds <- llm_config(
#   provider = "deepseek",
#   model    = "deepseek-chat"
# )
# 
# chat_ds <- chat_session(cfg_ds, system = "Be concise.")
# chat_ds$send("Say a warm hello in one short sentence.")
# chat_ds$send("Now say it in Esperanto.")

## -----------------------------------------------------------------------------
# cfg_reason <- llm_config(
#   provider = "deepseek",
#   model    = "deepseek-reasoner"
# )
# 
# chat_reason <- chat_session(cfg_reason, system = "Be concise.")
# chat_reason$send("Name one interesting fact about honey bees.")

## -----------------------------------------------------------------------------
# cfg_groq1 <- llm_config(
#   provider = "groq",
#   model    = "llama-3.1-8b-instant"
# )
# 
# chat_groq1 <- chat_session(cfg_groq1, system = "Be concise.")
# chat_groq1$send("Give me a single-sentence fun fact about volcanoes.")

## -----------------------------------------------------------------------------
# cfg_groq2 <- llm_config(
#   provider = "groq",
#   model    = "openai/gpt-oss-20b"
# )
# 
# chat_groq2 <- chat_session(cfg_groq2, system = "Be concise.")
# chat_groq2$send("Share a short fun fact about octopuses.")

## -----------------------------------------------------------------------------
# chat_ds$send("What did I ask you to do in my first message?")
# # The model can reference the earlier "Say a warm hello" request

## -----------------------------------------------------------------------------
# # View all messages
# as.data.frame(chat_ds)
# 
# # Get summary statistics
# summary(chat_ds)

## -----------------------------------------------------------------------------
# schema <- list(
#   type = "object",
#   properties = list(
#     answer     = list(type = "string"),
#     confidence = list(type = "number")
#   ),
#   required = list("answer", "confidence"),
#   additionalProperties = FALSE
# )
# 
# chat_ds$send_structured(
#   "Return an answer and a confidence score (0-1) about: Why is the sky blue?",
#   schema
# )

