% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/LLM_robust_utils.R
\name{cache_llm_call}
\alias{cache_llm_call}
\title{Cache LLM API Calls}
\usage{
cache_llm_call(config, messages, verbose = FALSE, json = FALSE)
}
\arguments{
\item{config}{An \code{llm_config} object from \code{\link{llm_config}}.}

\item{messages}{A list of message objects or character vector for embeddings.}

\item{verbose}{Logical. If TRUE, prints the full API response (passed to \code{\link{call_llm}}).}

\item{json}{Logical. If TRUE, returns raw JSON (passed to \code{\link{call_llm}}).}
}
\value{
The (memoised) response object from \code{\link{call_llm}}.
}
\description{
A memoised version of \code{\link{call_llm}} to avoid repeated identical requests.
}
\details{
\itemize{
\item Requires the \code{memoise} package. Add \code{memoise} to your
package's DESCRIPTION.
\item Clearing the cache can be done via \code{memoise::forget(cache_llm_call)}
or by restarting your R session.
}
}
\examples{
\dontrun{
  # Using cache_llm_call:
  response1 <- cache_llm_call(my_config, list(list(role="user", content="Hello!")))
  # Subsequent identical calls won't hit the API unless we clear the cache.
  response2 <- cache_llm_call(my_config, list(list(role="user", content="Hello!")))
}
}
