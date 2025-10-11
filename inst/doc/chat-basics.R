## -----------------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE, comment = "#>",
  eval = identical(tolower(Sys.getenv("LLMR_RUN_VIGNETTES", "false")), "true")
)

## -----------------------------------------------------------------------------
# library(LLMR)
# 
# cfg_openai <- llm_config(
#   provider = "openai",
#   model    = "gpt-5-nano",
# 
# )
# 
# chat_oai <- chat_session(cfg_openai, system = "Be concise.")
# chat_oai$send("Say a warm hello in one short sentence.")
# chat_oai$send("Now say it in Esperanto.")

## -----------------------------------------------------------------------------
# cfg_anthropic <- llm_config(
#   provider = "anthropic",
#   model    = "claude-sonnet-4-20250514",
#   max_tokens = 512   # avoid warnings; Anthropic requires max_tokens
# )
# 
# chat_claude <- chat_session(cfg_anthropic, system = "Be concise.")
# chat_claude$send("Name one interesting fact about honey bees.")

## -----------------------------------------------------------------------------
# cfg_gemini <- llm_config(
#   provider = "gemini",
#   model    = "gemini-2.5-flash-lite",
# 
# )
# 
# chat_gem <- chat_session(cfg_gemini, system = "Be concise.")
# chat_gem$send("Give me a single-sentence fun fact about volcanoes.")

## -----------------------------------------------------------------------------
# cfg_groq <- llm_config(
#   provider = "groq",
#   model    = "openai/gpt-oss-20b",
# 
# )
# 
# chat_groq <- chat_session(cfg_groq, system = "Be concise.")
# chat_groq$send("Share a short fun fact about octopuses.")

## -----------------------------------------------------------------------------
# chat_oai$send("What did I ask you to do in my first message?")
# # The model can reference the earlier "Say a warm hello" request

## -----------------------------------------------------------------------------
# # View all messages
# as.data.frame(chat_oai)
# 
# # Get summary statistics
# summary(chat_oai)

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
# chat_oai$send_structured(
#   "Return an answer and a confidence score (0-1) about: Why is the sky blue?",
#   schema
# )

