## -----------------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE, comment = "#>",
  eval = identical(tolower(Sys.getenv("LLMR_RUN_VIGNETTES", "false")), "true")
)

## -----------------------------------------------------------------------------
# library(LLMR)
# 
# cfg <- llm_config("groq", "openai/gpt-oss-20b", temperature = 0.2)
# 
# r <- call_llm(cfg, c(system = "Be concise.", user = "Capital of Mongolia?"))
# r                 # prints the text and a [model | finish | tokens | t] line
# as.character(r)   # just the text
# tokens(r)         # token counts as a list

## -----------------------------------------------------------------------------
# library(tibble)
# 
# reviews <- tibble(text = c("The food was cold.",
#                            "Absolutely loved it!",
#                            "It was fine, nothing special."))
# 
# reviews |>
#   llm_mutate(
#     sentiment = "Reply with one word (positive/negative/neutral): {text}",
#     .config   = cfg
#   )

## -----------------------------------------------------------------------------
# countries <- c("Mongolia", "Bolivia", "Chad")
# 
# llm_fn(countries,
#        prompt  = "Capital city of {x}. Reply with only the city name.",
#        .config = cfg)

## -----------------------------------------------------------------------------
# films <- tibble(title = c("Blade Runner", "Amelie", "Parasite", "Spirited Away"))
# 
# films |>
#   llm_mutate(
#     info       = "For the film {title}, give its director and release year.",
#     .config    = cfg,
#     .tags      = c("director", "year"),
#     .rows_per_prompt = 2
#   )

## -----------------------------------------------------------------------------
# emb_cfg <- llm_config("voyage", "voyage-3.5-lite", embedding = TRUE)
# 
# texts <- c("I love this restaurant.",
#            "The food was delicious.",
#            "My car broke down today.")
# 
# m <- get_batched_embeddings(texts, emb_cfg)
# dim(m)   # 3 texts x embedding dimension

## -----------------------------------------------------------------------------
# cosine <- function(a, b) sum(a * b) / sqrt(sum(a * a) * sum(b * b))
# 
# cosine(m[1, ], m[2, ])   # food vs food: high
# cosine(m[1, ], m[3, ])   # food vs car:  low

## -----------------------------------------------------------------------------
# llm_preview(reviews,
#             prompt  = "Reply with one word: {text}",
#             .config = cfg)

## -----------------------------------------------------------------------------
# out <- reviews |>
#   llm_mutate(sentiment = "One word for: {text}", .config = cfg)
# 
# llm_usage(out)
# llm_failures(out)

