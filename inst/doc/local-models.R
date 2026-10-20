## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE, comment = "#>",
  eval = identical(tolower(Sys.getenv("LLMR_RUN_VIGNETTES", "false")), "true") &&
    LLMR::llm_local_ready("qwen3-0.6b")
)

## -----------------------------------------------------------------------------
# library(LLMR)

## ----eval = FALSE-------------------------------------------------------------
# llm_local_install("qwen3-0.6b")

## -----------------------------------------------------------------------------
# llm_local_models()[, c("name", "kind", "size_mb", "license", "installed")]

## -----------------------------------------------------------------------------
# cfg <- llm_config("llamacpp", "qwen3-0.6b", temperature = 0, thinking = FALSE)
# call_llm(cfg, "What is the capital of France? Answer in one sentence.")

## -----------------------------------------------------------------------------
# llm_next_token(cfg, messages = "What is the capital of France? Answer in one sentence.",
#                prefill = "The capital of France is", top_n = 5)

## -----------------------------------------------------------------------------
# llm_continue(cfg, messages = "Is coffee good for you?",
#              prefill = "Honestly, I am not sure, but", n = 3,
#              temperature = 1, max_tokens = 30, seed = 110)[, c("sample", "continuation")]

## -----------------------------------------------------------------------------
# s <- llm_score(cfg, " Lyon.", prompt = "The capital of France is")
# s[, c("token", "logprob", "rank")]
# attr(s, "total_logprob")

## -----------------------------------------------------------------------------
# llm_choice_probs(cfg, c("Yes", "No"),
#                  messages = "Is Paris the capital of France? Answer yes or no.")

## -----------------------------------------------------------------------------
# llm_tokenize(cfg, "The capital of France is Paris.")
# cat(llm_render_template(cfg, "Name a color.", prefill = "My answer is"))

## ----eval = FALSE-------------------------------------------------------------
# llm_local_install("multilingual-e5-small")
# texts <- c("The government raised the national budget for schools.",
#            "دولت بودجهٔ ملی مدارس را افزایش داد.")
# X <- get_batched_embeddings(texts, llm_config("llamacpp", "multilingual-e5-small"))
# sum(X[1, ] * X[2, ])  # vectors are unit length, so this is the cosine

## ----eval = FALSE-------------------------------------------------------------
# llm_local_start("qwen3-0.6b", device = "cpu", threads = 4)

## ----eval = FALSE-------------------------------------------------------------
# llm_local_install(llm_local_model("Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf"))

## ----include = FALSE----------------------------------------------------------
# llm_local_stop()

