/**
 * JavaScript/TypeScript SDK for LLM Evaluation Platform External API
 * Provides a convenient interface for integrating with the evaluation platform.
 */

// Type definitions for TypeScript support
/**
 * @typedef {Object} EvaluationRequest
 * @property {string} user_input - User input text to evaluate
 * @property {string} model_output - Model response to evaluate
 * @property {string} model_name - Name of the model that generated the response
 * @property {string} [system_prompt] - System prompt used
 * @property {string[]} [criteria] - Evaluation criteria
 * @property {Object} [context] - Additional context
 * @property {string} [reference_answer] - Reference answer for comparison
 * @property {string} [session_id] - Session identifier
 * @property {Object} [metadata] - Additional metadata
 */

/**
 * @typedef {Object} EvaluationResponse
 * @property {string} evaluation_id - Unique evaluation identifier
 * @property {string} trace_id - Associated trace identifier
 * @property {number} overall_score - Overall evaluation score (0-1)
 * @property {Object<string, number>} criteria_scores - Scores for each criterion
 * @property {string} reasoning - Detailed reasoning for the evaluation
 * @property {number} confidence - Confidence in the evaluation (0-1)
 * @property {string} evaluator_model - Model used for evaluation
 * @property {number} evaluation_time_ms - Time taken for evaluation
 * @property {number} [cost_usd] - Cost of the evaluation
 */

/**
 * @typedef {Object} TraceFilter
 * @property {string[]} [model_names] - Filter by model names
 * @property {string} [date_from] - Start date (ISO format)
 * @property {string} [date_to] - End date (ISO format)
 * @property {number} [min_score] - Minimum evaluation score (0-1)
 * @property {number} [max_score] - Maximum evaluation score (0-1)
 * @property {string[]} [session_ids] - Filter by session IDs
 * @property {boolean} [has_evaluation] - Filter traces with/without evaluations
 * @property {number} [limit] - Maximum number of results (1-500)
 * @property {number} [offset] - Number of results to skip
 */

/**
 * @typedef {Object} BatchRequest
 * @property {string} operation - Batch operation type
 * @property {Object[]} items - Items to process
 * @property {Object} [options] - Operation options
 * @property {string} [callback_url] - Webhook URL for completion notification
 */

/**
 * Custom error classes for the SDK
 */
class LLMEvaluationAPIError extends Error {
    constructor(message, statusCode = null, responseData = null) {
        super(message);
        this.name = 'LLMEvaluationAPIError';
        this.statusCode = statusCode;
        this.responseData = responseData;
    }
}

class RateLimitError extends LLMEvaluationAPIError {
    constructor(message, retryAfter = 3600) {
        super(message, 429);
        this.name = 'RateLimitError';
        this.retryAfter = retryAfter;
    }
}

class AuthenticationError extends LLMEvaluationAPIError {
    constructor(message = 'Invalid API key') {
        super(message, 401);
        this.name = 'AuthenticationError';
    }
}

/**
 * Main client class for the LLM Evaluation Platform API
 */
class LLMEvaluationClient {
    /**
     * Initialize the evaluation client
     * @param {string} apiKey - Your API key for authentication
     * @param {string} [baseUrl='http://localhost:8000'] - Base URL of the evaluation platform
     * @param {number} [timeout=30000] - Request timeout in milliseconds
     * @param {number} [maxRetries=3] - Maximum number of retry attempts
     * @param {number} [retryDelay=1000] - Delay between retries in milliseconds
     */
    constructor(apiKey, baseUrl = 'http://localhost:8000', timeout = 30000, maxRetries = 3, retryDelay = 1000) {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.timeout = timeout;
        this.maxRetries = maxRetries;
        this.retryDelay = retryDelay;
        
        this.defaultHeaders = {
            'X-API-Key': apiKey,
            'Content-Type': 'application/json',
            'User-Agent': 'LLMEval-JS-SDK/1.0.0'
        };
    }

    /**
     * Make an HTTP request with retry logic
     * @private
     */
    async _makeRequest(method, endpoint, data = null, params = null) {
        const url = new URL(endpoint.replace(/^\//, ''), this.baseUrl + '/');
        
        if (params) {
            Object.keys(params).forEach(key => {
                if (params[key] != null) {
                    url.searchParams.append(key, params[key]);
                }
            });
        }

        const requestOptions = {
            method: method.toUpperCase(),
            headers: { ...this.defaultHeaders },
            signal: AbortSignal.timeout(this.timeout)
        };

        if (data && (method.toUpperCase() === 'POST' || method.toUpperCase() === 'PUT')) {
            requestOptions.body = JSON.stringify(data);
        }

        for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
            try {
                const response = await fetch(url.toString(), requestOptions);
                
                // Handle rate limiting
                if (response.status === 429) {
                    const retryAfter = parseInt(response.headers.get('Retry-After') || '3600');
                    throw new RateLimitError(
                        `Rate limit exceeded. Retry after ${retryAfter} seconds.`,
                        retryAfter
                    );
                }

                // Handle authentication errors
                if (response.status === 401) {
                    throw new AuthenticationError('Invalid API key or expired token');
                }

                // Handle other client errors
                if (response.status >= 400 && response.status < 500) {
                    const errorData = response.headers.get('content-type')?.includes('application/json')
                        ? await response.json()
                        : { detail: await response.text() };
                    
                    throw new LLMEvaluationAPIError(
                        `Client error: ${response.status} - ${errorData.detail || response.statusText}`,
                        response.status,
                        errorData
                    );
                }

                // Handle server errors with retry
                if (response.status >= 500) {
                    if (attempt < this.maxRetries) {
                        await this._sleep(this.retryDelay * Math.pow(2, attempt)); // Exponential backoff
                        continue;
                    }

                    const errorData = response.headers.get('content-type')?.includes('application/json')
                        ? await response.json()
                        : { detail: await response.text() };
                    
                    throw new LLMEvaluationAPIError(
                        `Server error: ${response.status} - ${errorData.detail || response.statusText}`,
                        response.status,
                        errorData
                    );
                }

                // Success response
                if (!response.ok) {
                    throw new LLMEvaluationAPIError(`HTTP error: ${response.status} ${response.statusText}`);
                }

                return await response.json();

            } catch (error) {
                if (error instanceof LLMEvaluationAPIError) {
                    throw error;
                }

                if (attempt < this.maxRetries && !error.name?.includes('AbortError')) {
                    await this._sleep(this.retryDelay * Math.pow(2, attempt));
                    continue;
                }

                throw new LLMEvaluationAPIError(`Request failed: ${error.message}`);
            }
        }

        throw new LLMEvaluationAPIError('Max retries exceeded');
    }

    /**
     * Sleep utility for retry delays
     * @private
     */
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Check API health status
     * @returns {Promise<Object>} Health status response
     */
    async healthCheck() {
        return this._makeRequest('GET', '/api/external/health');
    }

    /**
     * Get list of available evaluation models and capabilities
     * @returns {Promise<Object>} Available models and capabilities
     */
    async getAvailableModels() {
        return this._makeRequest('GET', '/api/external/models');
    }

    /**
     * Create a new evaluation for an input-output pair
     * @param {EvaluationRequest} request - Evaluation request data
     * @returns {Promise<EvaluationResponse>} Evaluation response with scores and reasoning
     */
    async createEvaluation(request) {
        // Remove null/undefined values
        const cleanRequest = Object.fromEntries(
            Object.entries(request).filter(([_, v]) => v != null)
        );

        return this._makeRequest('POST', '/api/external/evaluations', cleanRequest);
    }

    /**
     * List evaluations with optional filtering
     * @param {Object} options - Query options
     * @param {number} [options.limit=50] - Maximum number of results (1-500)
     * @param {number} [options.offset=0] - Number of results to skip
     * @param {string} [options.modelName] - Filter by model name
     * @param {number} [options.minScore] - Minimum evaluation score (0-1)
     * @returns {Promise<EvaluationResponse[]>} List of evaluation responses
     */
    async listEvaluations({ limit = 50, offset = 0, modelName = null, minScore = null } = {}) {
        const params = { limit, offset };
        if (modelName) params.model_name = modelName;
        if (minScore != null) params.min_score = minScore;

        return this._makeRequest('GET', '/api/external/evaluations', null, params);
    }

    /**
     * Submit a trace for evaluation
     * @param {Object} traceData - Trace data including user_input, model_output, model_name
     * @returns {Promise<Object>} Trace submission response
     */
    async submitTrace(traceData) {
        return this._makeRequest('POST', '/api/external/traces', traceData);
    }

    /**
     * List traces with optional filtering
     * @param {TraceFilter} [filterParams] - Filter parameters for traces
     * @returns {Promise<Object[]>} List of trace responses
     */
    async listTraces(filterParams = {}) {
        // Remove null/undefined values
        const params = Object.fromEntries(
            Object.entries(filterParams).filter(([_, v]) => v != null)
        );

        return this._makeRequest('GET', '/api/external/traces', null, params);
    }

    /**
     * Create a batch operation for processing multiple items
     * @param {BatchRequest} batchRequest - Batch operation request
     * @returns {Promise<Object>} Batch operation response with job ID and status
     */
    async createBatchOperation(batchRequest) {
        const cleanRequest = Object.fromEntries(
            Object.entries(batchRequest).filter(([_, v]) => v != null)
        );

        return this._makeRequest('POST', '/api/external/batch', cleanRequest);
    }

    /**
     * Get API usage statistics for the current API key
     * @returns {Promise<Object>} Usage statistics
     */
    async getUsageStatistics() {
        return this._makeRequest('GET', '/api/external/usage');
    }

    /**
     * Convenience method for evaluating a single text response
     * @param {string} userInput - The user's input/question
     * @param {string} modelOutput - The model's response
     * @param {string} modelName - Name of the model that generated the response
     * @param {string[]} [criteria=['relevance', 'coherence']] - List of evaluation criteria
     * @param {Object} [options={}] - Additional options for EvaluationRequest
     * @returns {Promise<EvaluationResponse>} Evaluation response
     */
    async evaluateText(userInput, modelOutput, modelName, criteria = ['relevance', 'coherence'], options = {}) {
        const request = {
            user_input: userInput,
            model_output: modelOutput,
            model_name: modelName,
            criteria,
            ...options
        };

        return this.createEvaluation(request);
    }

    /**
     * Evaluate multiple input-output pairs in batch
     * @param {EvaluationRequest[]} evaluations - List of evaluation requests
     * @param {number} [batchSize=10] - Number of evaluations per batch
     * @param {string} [callbackUrl] - Optional webhook URL for completion notification
     * @returns {Promise<Object>} Batch operation response
     */
    async bulkEvaluate(evaluations, batchSize = 10, callbackUrl = null) {
        const batchRequest = {
            operation: 'evaluate',
            items: evaluations,
            options: { batch_size: batchSize }
        };

        if (callbackUrl) {
            batchRequest.callback_url = callbackUrl;
        }

        return this.createBatchOperation(batchRequest);
    }

    /**
     * Stream evaluation results as they complete
     * @param {EvaluationRequest[]} evaluations - List of evaluation requests
     * @param {number} [maxConcurrent=5] - Maximum concurrent evaluations
     * @returns {AsyncGenerator<Object>} Stream of evaluation results
     */
    async* streamEvaluations(evaluations, maxConcurrent = 5) {
        const semaphore = new Semaphore(maxConcurrent);
        
        const evaluateOne = async (evalRequest) => {
            await semaphore.acquire();
            try {
                return await this.createEvaluation(evalRequest);
            } finally {
                semaphore.release();
            }
        };

        // Create promises for all evaluations
        const promises = evaluations.map(evalRequest => evaluateOne(evalRequest));

        // Yield results as they complete
        const results = [];
        const pending = [...promises];

        while (pending.length > 0) {
            try {
                const result = await Promise.race(pending);
                const index = promises.indexOf(Promise.resolve(result));
                if (index !== -1) {
                    pending.splice(pending.indexOf(promises[index]), 1);
                }
                yield result;
            } catch (error) {
                yield { error: error.message, timestamp: new Date().toISOString() };
            }
        }
    }
}

/**
 * Simple semaphore implementation for concurrency control
 */
class Semaphore {
    constructor(limit) {
        this.limit = limit;
        this.current = 0;
        this.queue = [];
    }

    async acquire() {
        return new Promise((resolve) => {
            if (this.current < this.limit) {
                this.current++;
                resolve();
            } else {
                this.queue.push(resolve);
            }
        });
    }

    release() {
        this.current--;
        if (this.queue.length > 0) {
            this.current++;
            const resolve = this.queue.shift();
            resolve();
        }
    }
}

/**
 * Convenience function for quick evaluation of a single response
 * @param {string} apiKey - Your API key
 * @param {string} userInput - The user's input/question
 * @param {string} modelOutput - The model's response
 * @param {string} modelName - Name of the model
 * @param {string[]} [criteria] - Evaluation criteria
 * @param {string} [baseUrl='http://localhost:8000'] - API base URL
 * @returns {Promise<EvaluationResponse>} Evaluation result
 */
async function evaluateResponse(apiKey, userInput, modelOutput, modelName, criteria = null, baseUrl = 'http://localhost:8000') {
    const client = new LLMEvaluationClient(apiKey, baseUrl);
    return client.evaluateText(userInput, modelOutput, modelName, criteria);
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    // CommonJS
    module.exports = {
        LLMEvaluationClient,
        LLMEvaluationAPIError,
        RateLimitError,
        AuthenticationError,
        evaluateResponse
    };
} else if (typeof window !== 'undefined') {
    // Browser global
    window.LLMEval = {
        LLMEvaluationClient,
        LLMEvaluationAPIError,
        RateLimitError,
        AuthenticationError,
        evaluateResponse
    };
}

// Example usage
if (typeof window === 'undefined' && require.main === module) {
    // Node.js example
    (async () => {
        const client = new LLMEvaluationClient('your-api-key-here');
        
        try {
            // Check health
            const health = await client.healthCheck();
            console.log(`API Status: ${health.status}`);
            
            // Evaluate a response
            const result = await client.evaluateText(
                'What is the capital of France?',
                'The capital of France is Paris.',
                'gpt-4',
                ['accuracy', 'completeness']
            );
            
            console.log(`Evaluation Score: ${result.overall_score}`);
            console.log(`Reasoning: ${result.reasoning}`);
            
        } catch (error) {
            console.error('Error:', error.message);
            if (error instanceof RateLimitError) {
                console.log(`Rate limit exceeded. Retry after ${error.retryAfter} seconds.`);
            }
        }
    })();
} 