% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/LLM_parallel_utils.R
\name{call_llm_compare}
\alias{call_llm_compare}
\title{Parallel API calls: Multiple Configs, Fixed Message}
\usage{
call_llm_compare(configs_list, messages, ...)
}
\arguments{
\item{configs_list}{A list of llm_config objects to compare.}

\item{messages}{A character vector or a list of message objects (same for all configs).}

\item{...}{Additional arguments passed to \code{call_llm_par} (e.g., tries, verbose, progress).}
}
\value{
A tibble with columns: config_index (metadata), provider, model,
all varying model parameters, response_text, raw_response_json, success, error_message.
}
\description{
Compares different configurations (models, providers, settings) using the same message.
Perfect for benchmarking across different models or providers.
This function requires setting up the parallel environment using \code{setup_llm_parallel}.
}
\section{Parallel Workflow}{

All parallel functions require the \code{future} backend to be configured.
The recommended workflow is:
\enumerate{
\item Call \code{setup_llm_parallel()} once at the start of your script.
\item Run one or more parallel experiments (e.g., \code{call_llm_broadcast()}).
\item Call \code{reset_llm_parallel()} at the end to restore sequential processing.
}
}

\examples{
\dontrun{
  # Compare different models
  config1 <- llm_config(provider = "openai", model = "gpt-4o-mini",
                        api_key = Sys.getenv("OPENAI_API_KEY"))
  config2 <- llm_config(provider = "openai", model = "gpt-4.1-nano",
                        api_key = Sys.getenv("OPENAI_API_KEY"))

  configs_list <- list(config1, config2)
  messages <- "Explain quantum computing"

  setup_llm_parallel(workers = 4, verbose = TRUE)
  results <- call_llm_compare(configs_list, messages)
  reset_llm_parallel(verbose = TRUE)
}
}
\seealso{
\code{\link{setup_llm_parallel}}, \code{\link{reset_llm_parallel}}
}
