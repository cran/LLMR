
# ----- Helper Functions -----

#' Perform API Request
#'
#' Internal helper function to perform the API request and process the response.
#'
#' @keywords internal
#' @noRd
#' @importFrom httr2 req_perform resp_body_raw resp_body_json
#' @importFrom rlang `%||%`
perform_request <- function(req, verbose, json) {
  resp <- httr2::req_perform(req)

  # Get the raw response as text
  raw_response <- httr2::resp_body_raw(resp)
  raw_json <- rawToChar(raw_response)

  # Parse the response as JSON
  content <- httr2::resp_body_json(resp)

  if (verbose) {
    cat("Full API Response:\n")
    print(content)
  }

  if (json) {
    # Return the text with attributes for full_response and raw_json
    text <- extract_text(content)
    attr(text, "full_response") <- content
    attr(text, "raw_json") <- raw_json
    return(text)
  }

  # Extract the text response
  text <- extract_text(content)
  attr(text, "full_response") <- content
  return(text)
}

#' Extract Text from API Response
#'
#' Internal helper function to extract text from the API response content.
#'
#' @keywords internal
#' @noRd
extract_text <- function(content) {
  if (is.null(content$choices) && is.null(content$content)) {
    stop("No recognizable content returned from API.")
  }

  if (!is.null(content$choices)) {
    # For APIs like OpenAI, Groq, Together AI
    if (length(content$choices) == 0 || is.null(content$choices[[1]]$message$content)) {
      stop("No choices returned from API.")
    }
    return(content$choices[[1]]$message$content)
  }

  if (!is.null(content$content)) {
    # For Anthropic
    if (length(content$content) == 0 || is.null(content$content[[1]]$text)) {
      stop("No content returned from Anthropic API.")
    }
    return(content$content[[1]]$text)
  }

  stop("Unable to extract text from the API response.")
}

#' Format Anthropic Messages
#'
#' Internal helper function to format messages for Anthropic API.
#'
#' @keywords internal
#' @noRd
format_anthropic_messages <- function(messages) {
  # Separate system messages from user/assistant messages
  system_messages <- purrr::keep(messages, ~ .x$role == "system")
  user_messages <- purrr::keep(messages, ~ .x$role != "system")

  # Concatenate system messages if any
  if (length(system_messages) > 0) {
    system_text <- paste(sapply(system_messages, function(x) x$content), collapse = " ")
  } else {
    system_text <- NULL
  }

  # Convert user messages to Anthropic's expected format
  formatted_messages <- lapply(user_messages, function(msg) {
    list(
      role = msg$role,
      content = list(
        list(
          type = "text",
          text = msg$content
        )
      )
    )
  })

  list(system_text = system_text, formatted_messages = formatted_messages)
}

# ----- Exported Functions -----
#' Create LLM Configuration
#'
#' Creates a configuration object for interacting with a specified LLM API provider.
#'
#' @param provider A string specifying the API provider. Supported providers include:
#'   "openai" for OpenAI,
#'   "anthropic" for Anthropic,
#'   "groq" for Groq,
#'   "together" for Together AI;
#'   This configuration is extendable; to add support for additional providers, define a new S3 method for `call_llm` corresponding to the provider.
#' @param model The model name to use. This depends on the provider. For example:
#'   OpenAI: "gpt-4o-mini",
#'   Anthropic: "claude-3-opus-20240229",
#'   Groq: "llama-3.1-8b-instant",
#'   Together AI: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".
#' @param api_key Your API key for the provider.
#' @param ... Additional model-specific parameters (e.g., `temperature`, `max_tokens`).
#'
#' @return An object of class `llm_config` containing API and model parameters.
#' @export
#'
#' @examples
#' \dontrun{
#'   # Obtain API keys from:
#'   # OpenAI: https://platform.openai.com/api-keys
#'   # Groq: https://console.groq.com/keys
#'   # Anthropic: https://console.anthropic.com/settings/keys
#'   # Together AI: https://api.together.ai/settings/api-keys
#'
#'   # ----- OpenAI Example -----
#'   openai_config <- llm_config(
#'     provider = "openai",
#'     model = "gpt-4o-mini",
#'     api_key = OPENAI_KEY,
#'     temperature = 0.7,
#'     max_tokens = 500
#'   )
#'
#'   # ----- Anthropic Example -----
#'   anthropic_config <- llm_config(
#'     provider = "anthropic",
#'     model = "claude-3-opus-20240229",
#'     api_key = ANTHROPIC_API_KEY,
#'     max_tokens = 500
#'   )
#'
#'   # ----- Groq Example -----
#'   groq_config <- llm_config(
#'     provider = "groq",
#'     model = "llama-3.1-8b-instant",
#'     api_key = GROQ_API_KEY,
#'     temperature = 0.3,
#'     max_tokens = 1000
#'   )
#'
#'   # ----- Together AI Example -----
#'   together_config <- llm_config(
#'     provider = "together",
#'     model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#'     api_key = TOGETHER_KEY,
#'     temperature = 0.5,
#'     max_tokens = 1000
#'   )
#'
#'   # This configuration is extendable by defining new `call_llm` methods for additional providers.
#' }
#'
#' @seealso \code{\link{call_llm}}

llm_config <- function(provider, model, api_key, ...) {
  model_params <- list(...)
  config <- list(
    provider = provider,
    model = model,
    api_key = api_key,
    model_params = model_params
  )
  class(config) <- c("llm_config", provider)
  return(config)
}


#' Call LLM API
#'
#' Sends a message to the specified LLM API and retrieves the response.
#'
#' @param config An `llm_config` object created by `llm_config()`.
#' @param messages A list of message objects to send to the API.
#' @param verbose Logical. If `TRUE`, prints the full API response.
#' @param json Logical. If `TRUE`, returns the raw JSON response as an attribute.
#'
#' @return The generated text response with additional attributes based on parameters.
#' @export
#'
#' @examples
#' \dontrun{
#'   # ----- Groq Example -----
#'   groq_config <- llm_config(
#'     provider = "groq",
#'     model = "llama-3.1-8b-instant",
#'     api_key = Sys.getenv("GROQ_KEY"),
#'     temperature = 0.7,
#'     max_tokens = 500
#'   )
#'
#'   # Define the message with a system prompt
#'   message <- list(
#'     list(role = "system", content = "You ONLY fill in the blank. Do NOT answer in full sentences."),
#'     list(role = "user", content = "What's the capital of France? ----")
#'   )
#'
#'   # Call the LLM
#'   response <- call_llm(groq_config, message)
#'
#'   # Display the response
#'   cat("Groq Response:", response, "\n")
#'
#'   # Extract and print the full API response
#'   full.response <- attr(response, which = 'full_response')
#'   print(full.response)
#'
#'
#'   # -----  OpenAI Example with More Parameters -----
#'   # Create a configuration with more parameters
#'   comprehensive_openai_config <- llm_config(
#'     provider = "openai",
#'     model = "gpt-4o-mini",
#'     api_key = Sys.getenv("OPENAI_KEY"),
#'     temperature = 1,          # Controls the randomness of the output
#'     max_tokens = 750,            # Maximum number of tokens to generate
#'     top_p = 1,                 # Nucleus sampling parameter
#'     frequency_penalty = 0.5,     # Penalizes new tokens based on their frequency
#'     presence_penalty = 0.3        # Penalizes new tokens based on their presence
#'   )
#'
#'   # Define a more complex message
#'   comprehensive_message <- list(
#'     list(role = "system", content = "You are an expert data scientist."),
#'     list(role = "user", content = "When will you ever use OLS?")
#'   )
#'
#'   # Call the LLM with all parameters
#'   comprehensive_response <- call_llm(
#'     config = comprehensive_openai_config,
#'     messages = comprehensive_message,
#'     json = TRUE        # Retrieve the raw JSON response as an attribute
#'   )
#'
#'   # Display the generated text response
#'   cat("Comprehensive OpenAI Response:", comprehensive_response, "\n")
#'
#'   # Access and print the raw JSON response
#'   raw_json_response <- attr(comprehensive_response, "raw_json")
#'   print(raw_json_response)
#' }
#'
#'
#' @seealso \code{\link{llm_config}}
call_llm <- function(config, messages, verbose = FALSE, json = FALSE) {
  UseMethod("call_llm", config)
}

#' @export
call_llm.default <- function(config, messages, verbose = FALSE, json = FALSE) {
  stop(paste("Unsupported API provider:", config$provider))
}

#' @export
call_llm.openai <- function(config, messages, verbose = FALSE, json = FALSE) {
  body <- list(
    model = config$model,
    messages = messages,
    temperature = rlang::`%||%`(config$model_params$temperature, 1),
    max_tokens = rlang::`%||%`(config$model_params$max_tokens, 1024),
    top_p = rlang::`%||%`(config$model_params$top_p, 1),
    frequency_penalty = rlang::`%||%`(config$model_params$frequency_penalty, 0),
    presence_penalty = rlang::`%||%`(config$model_params$presence_penalty, 0)
  )

  req <- httr2::request("https://api.openai.com/v1/chat/completions") |>
    httr2::req_headers(
      "Content-Type" = "application/json",
      "Authorization" = paste("Bearer", config$api_key)
    ) |>
    httr2::req_body_json(body)

  perform_request(req, verbose, json)
}

#' @export
call_llm.anthropic <- function(config, messages, verbose = FALSE, json = FALSE) {
  formatted <- format_anthropic_messages(messages)

  body <- list(
    model = config$model,
    max_tokens = rlang::`%||%`(config$model_params$max_tokens, 1024),
    temperature = rlang::`%||%`(config$model_params$temperature, 1)
  )

  if (!is.null(formatted$system_text)) {
    body$system <- formatted$system_text
  }

  if (length(formatted$formatted_messages) > 0) {
    body$messages <- formatted$formatted_messages
  }

  req <- httr2::request("https://api.anthropic.com/v1/messages") |>
    httr2::req_headers(
      "Content-Type" = "application/json",
      "x-api-key" = config$api_key,
      "anthropic-version" = "2023-06-01"
    ) |>
    httr2::req_body_json(body)

  perform_request(req, verbose, json)
}

#' @export
call_llm.groq <- function(config, messages, verbose = FALSE, json = FALSE) {
  body <- list(
    model = config$model,
    messages = messages,
    temperature = rlang::`%||%`(config$model_params$temperature, 0.7),
    max_tokens = rlang::`%||%`(config$model_params$max_tokens, 1024)
  )

  req <- httr2::request("https://api.groq.com/openai/v1/chat/completions") |>
    httr2::req_headers(
      "Content-Type" = "application/json",
      "Authorization" = paste("Bearer", config$api_key)
    ) |>
    httr2::req_body_json(body)

  perform_request(req, verbose, json)
}

#' @export
call_llm.together <- function(config, messages, verbose = FALSE, json = FALSE) {
  body <- list(
    model = config$model,
    messages = messages,
    max_tokens = rlang::`%||%`(config$model_params$max_tokens, 1024),
    temperature = rlang::`%||%`(config$model_params$temperature, 0.7),
    top_p = rlang::`%||%`(config$model_params$top_p, 0.7),
    top_k = rlang::`%||%`(config$model_params$top_k, 50),
    repetition_penalty = rlang::`%||%`(config$model_params$repetition_penalty, 1),
    stop = rlang::`%||%`(config$model_params$stop, c("<|eot_id|>", "<|eom_id|>")),
    stream = rlang::`%||%`(config$model_params$stream, FALSE)
  )

  req <- httr2::request("https://api.together.xyz/v1/chat/completions") |>
    httr2::req_headers(
      "Content-Type" = "application/json",
      "Authorization" = paste("Bearer", config$api_key)
    ) |>
    httr2::req_body_json(body)

  perform_request(req, verbose, json)
}
