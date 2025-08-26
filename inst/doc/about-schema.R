## -----------------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE, comment = "#>",
  eval = identical(tolower(Sys.getenv("LLMR_RUN_VIGNETTES", "false")), "true") )

## -----------------------------------------------------------------------------
safe <- function(expr) tryCatch(expr, error = function(e) {message("ERROR: ", e$message); NULL})

## -----------------------------------------------------------------------------
safe({
  library(LLMR)
  cfg <- llm_config(
    provider = "openai",                # try "groq" or "together" too
    model    = "gpt-4o-mini",
    temperature = 0
  )

  # Flip JSON mode on (OpenAI-compat shape)
  cfg_json <- enable_structured_output(cfg, schema = NULL)

  res    <- call_llm(cfg_json, 'Give me a JSON object {"ok": true, "n": 3}.')
  parsed <- llm_parse_structured(res)

  cat("Raw text:\n", as.character(res), "\n\n")
  str(parsed)
})

## -----------------------------------------------------------------------------
safe({
  library(LLMR); library(dplyr)

  # Schema: make every property required to satisfy Groq's stricter check
  schema <- list(
    type = "object",
    additionalProperties = FALSE,
    properties = list(
      title = list(type = "string"),
      year  = list(type = "integer"),
      tags  = list(type = "array", items = list(type = "string"))
    ),
    required = list("title","year","tags")
  )

  cfg <- llm_config(
    provider = "groq",
    model    = "qwen-2.5-72b-instruct",   # a Qwen Instruct model on Groq
    temperature = 0
  )
  cfg_strict <- enable_structured_output(cfg, schema = schema, strict = TRUE)

  df  <- tibble(x = c("BERT paper", "Vision Transformers"))
  out <- llm_fn_structured(
    df,
    prompt   = "Return JSON about '{x}' with fields title, year, tags.",
    .config  = cfg_strict,
    .schema  = schema,          # send schema to provider
    .fields  = c("title","year","tags"),
    .validate_local = TRUE
  )

  out %>% select(structured_ok, structured_valid, title, year, tags) %>% print(n = Inf)
})

## -----------------------------------------------------------------------------
safe({
  library(LLMR)
  schema <- list(
    type="object",
    properties=list(answer=list(type="string"), confidence=list(type="number")),
    required=list("answer","confidence"),
    additionalProperties=FALSE
  )

  cfg <- llm_config("anthropic","claude-3-7", temperature = 0)
  cfg <- enable_structured_output(cfg, schema = schema, name = "llmr_schema")

  res <- call_llm(cfg, c(
    system = "Return only the tool result that matches the schema.",
    user   = "Answer: capital of Japan; include confidence in [0,1]."
  ))

  parsed <- llm_parse_structured(res)
  str(parsed)
})

## -----------------------------------------------------------------------------
safe({
  library(LLMR)

  cfg <- llm_config(
    "gemini", "gemini-2.0-flash",
    response_mime_type = "application/json"  # ask for JSON back
    # Optionally: gemini_enable_response_schema = TRUE, response_schema = <your JSON Schema>
  )

  res <- call_llm(cfg, c(
    system = "Reply as JSON only.",
    user   = "Produce fields name and score about 'MNIST'."
  ))
  str(llm_parse_structured(res))
})

## -----------------------------------------------------------------------------
safe({
  library(LLMR); library(tibble)

  messy <- c(
    '```json\n{"x": 1, "y": [1,2,3]}\n```',
    'Sure! Here is JSON: {"x":"1","y":"oops"} trailing words',
    '{"x":1, "y":[2,3,4]}'
  )

  tibble(response_text = messy) |>
    llm_parse_structured_col(
      fields = c(x = "x", y = "/y/0")   # dot/bracket or JSON Pointer
    ) |>
    print(n = Inf)
})

