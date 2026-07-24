## -----------------------------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE, comment = "#>",
  eval = identical(tolower(Sys.getenv("LLMR_RUN_VIGNETTES", "false")), "true")
)

## ----setup--------------------------------------------------------------------
# library(LLMR)

## ----tool-def-----------------------------------------------------------------
# survey <- data.frame(
#   group   = rep(c("treatment", "control"), each = 4),
#   support = c(6, 7, 5, 7, 4, 3, 5, 4)
# )
# 
# group_stats <- llm_tool(
#   function(group) {
#     rows <- survey[survey$group == group, ]
#     if (!nrow(rows)) return(paste0("No group called ", group))
#     sprintf("n = %d, mean support = %.2f", nrow(rows), mean(rows$support))
#   },
#   name        = "group_stats",
#   description = "Sample size and mean support (1-7 scale) for one experimental group.",
#   parameters  = list(group = list(type = "string",
#                                   description = "Group name: treatment or control"))
# )

## ----tool-loop----------------------------------------------------------------
# cfg <- llm_config("groq", "openai/gpt-oss-20b", temperature = 0)
# 
# r <- call_llm_tools(
#   cfg,
#   "Which group reports higher support, and by how much? Use the tool.",
#   tools = group_stats
# )
# r

## ----tool-history-------------------------------------------------------------
# r$tool_history

## ----tool-loop-spend----------------------------------------------------------
# r$tool_loop

## ----stream-basic-------------------------------------------------------------
# r <- call_llm_stream(cfg, "In two sentences: why do surveys weight responses?")
# tokens(r)

## ----stream-callback----------------------------------------------------------
# seen <- character(0)
# r <- call_llm_stream(cfg, "Count from one to five, words only.",
#                      callback = function(chunk) seen <<- c(seen, chunk))
# length(seen)        # the reply arrived in this many pieces
# as.character(r)     # and assembled into the usual llmr_response

## ----logprobs-----------------------------------------------------------------
# cfg_lp <- llm_config("deepseek", "deepseek-chat", temperature = 0,
#                      logprobs = TRUE, top_logprobs = 5, max_tokens = 4)
# 
# r <- call_llm(cfg_lp, c(
#   system = "Classify the sentiment of the review. Reply with exactly one word: positive or negative.",
#   user   = "The plot was predictable, but I cried at the end."))
# 
# lp <- llm_logprobs(r)
# data.frame(token = lp$token, p = exp(lp$logprob))

## ----logprobs-alts------------------------------------------------------------
# alts <- lp$top_logprobs[[1]]
# transform(alts, p = exp(logprob))[, c("token", "p")]

