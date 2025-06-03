/**
 * LLM Evaluation Platform JavaScript/TypeScript SDK
 * 
 * This SDK provides easy integration with the LLM Evaluation Platform
 * for both browser and Node.js environments.
 * 
 * Installation:
 *   npm install axios
 * 
 * Usage:
 *   import { LLMEvalClient } from './llm-eval-sdk';
 *   
 *   const client = new LLMEvalClient('http://localhost:8000');
 *   
 *   await client.sendTrace({
 *     traceId: 'unique_id',
 *     modelName: 'gpt-4',
 *     userQuery: 'What is AI?',
 *     aiResponse: 'AI is artificial intelligence...',
 *     metadata: { temperature: 0.7 }
 *   });
 */

// For Node.js environments
const axios = typeof window === 'undefined' ? require('axios') : null;

/**
 * @typedef {Object} TraceData
 * @property {string} traceId - Unique identifier for the trace
 * @property {string} modelName - Name of the AI model
 * @property {string} userQuery - User's input query
 * @property {string} aiResponse - AI model's response
 * @property {string} [systemPrompt] - Optional system prompt
 * @property {Array} [functionsCalled] - Optional list of function calls
 * @property {Object} [metadata] - Optional metadata object
 * @property {number} [tokensUsed] - Optional token count
 * @property {number} [responseTimeMs] - Optional response time in milliseconds
 * @property {number} [cost] - Optional cost in dollars
 * @property {string} [timestamp] - Optional timestamp (ISO string)
 */

/**
 * @typedef {Object} EvaluationData
 * @property {string} traceId - ID of the trace to evaluate
 * @property {string} status - 'accepted' or 'rejected'
 * @property {string} [reason] - Optional reason for rejection
 * @property {string} [notes] - Optional evaluation notes
 */

class LLMEvalClient {
    /**
     * Initialize the LLM Evaluation client
     * @param {string} baseUrl - Base URL of the evaluation platform
     * @param {string} [apiKey] - Optional API key for authentication
     * @param {number} [timeout=30000] - Request timeout in milliseconds
     */
    constructor(baseUrl, apiKey = null, timeout = 30000) {
        this.baseUrl = baseUrl.replace(/\/+$/, ''); // Remove trailing slashes
        this.apiKey = apiKey;
        this.timeout = timeout;
        
        // Setup HTTP client
        if (typeof window !== 'undefined') {
            // Browser environment
            this.httpClient = window.fetch ? this : axios;
        } else {
            // Node.js environment
            this.httpClient = axios.create({
                baseURL: this.baseUrl,
                timeout: this.timeout,
                headers: {
                    'Content-Type': 'application/json',
                    'User-Agent': 'LLMEvalSDK-JS/1.0'
                }
            });
            
            if (apiKey) {
                this.httpClient.defaults.headers.common['Authorization'] = `Bearer ${apiKey}`;
            }
        }
    }
    
    /**
     * Make HTTP request (browser-compatible)
     * @private
     */
    async _makeRequest(method, url, data = null) {
        const fullUrl = this.baseUrl + url;
        const headers = {
            'Content-Type': 'application/json',
        };
        
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        
        const config = {
            method: method.toUpperCase(),
            headers: headers,
        };
        
        if (data) {
            config.body = JSON.stringify(data);
        }
        
        try {
            let response;
            
            if (typeof window !== 'undefined' && window.fetch) {
                // Browser fetch API
                response = await fetch(fullUrl, config);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                return await response.json();
            } else {
                // Node.js axios
                const axiosConfig = {
                    method: method,
                    url: url,
                    timeout: this.timeout
                };
                
                if (data) {
                    axiosConfig.data = data;
                }
                
                response = await this.httpClient(axiosConfig);
                return response.data;
            }
        } catch (error) {
            console.error(`Failed to ${method} ${url}:`, error);
            throw error;
        }
    }
    
    /**
     * Send a single trace to the evaluation platform
     * @param {TraceData} traceData - Trace data to send
     * @returns {Promise<Object>} Response from the webhook endpoint
     */
    async sendTrace(traceData) {
        // Add timestamp if not provided
        if (!traceData.timestamp) {
            traceData.timestamp = new Date().toISOString();
        }
        
        return await this._makeRequest('POST', '/webhook/trace', traceData);
    }
    
    /**
     * Send multiple traces in a single batch request
     * @param {TraceData[]} traces - Array of trace data
     * @param {string} [source='javascript_sdk'] - Source system identifier
     * @param {string} [batchId] - Optional batch identifier
     * @returns {Promise<Object>} Response from the batch webhook endpoint
     */
    async sendTracesBatch(traces, source = 'javascript_sdk', batchId = null) {
        // Add timestamps to traces if not provided
        const tracesWithTimestamps = traces.map(trace => ({
            ...trace,
            timestamp: trace.timestamp || new Date().toISOString()
        }));
        
        const payload = {
            traces: tracesWithTimestamps,
            source: source,
            batchId: batchId
        };
        
        return await this._makeRequest('POST', '/webhook/batch', payload);
    }
    
    /**
     * Get traces from the platform
     * @param {number} [limit=50] - Maximum number of traces to return
     * @param {string} [status] - Optional status filter
     * @returns {Promise<Array>} List of trace data
     */
    async getTraces(limit = 50, status = null) {
        let url = `/api/traces?limit=${limit}`;
        if (status) {
            url += `&status=${encodeURIComponent(status)}`;
        }
        
        return await this._makeRequest('GET', url);
    }
    
    /**
     * Get a specific trace by ID
     * @param {string} traceId - Trace ID
     * @returns {Promise<Object>} Trace data
     */
    async getTrace(traceId) {
        return await this._makeRequest('GET', `/api/traces/${encodeURIComponent(traceId)}`);
    }
    
    /**
     * Evaluate a trace (accept/reject)
     * @param {EvaluationData} evaluationData - Evaluation data
     * @returns {Promise<Object>} Evaluation response
     */
    async evaluateTrace(evaluationData) {
        return await this._makeRequest('POST', '/api/evaluations', evaluationData);
    }
    
    /**
     * Get platform statistics
     * @returns {Promise<Object>} Platform stats
     */
    async getStats() {
        return await this._makeRequest('GET', '/webhook/stats');
    }
    
    /**
     * Check platform health
     * @returns {Promise<Object>} Health status
     */
    async healthCheck() {
        return await this._makeRequest('GET', '/health');
    }
}

/**
 * Real-time streaming client using Server-Sent Events
 */
class LLMEvalStreamClient {
    /**
     * Initialize the streaming client
     * @param {string} baseUrl - Base URL of the evaluation platform
     * @param {string} [apiKey] - Optional API key for authentication
     */
    constructor(baseUrl, apiKey = null) {
        this.baseUrl = baseUrl.replace(/\/+$/, '');
        this.apiKey = apiKey;
        this.eventSources = new Map();
    }
    
    /**
     * Create EventSource with proper headers
     * @private
     */
    _createEventSource(url) {
        // Note: EventSource doesn't support custom headers in browsers
        // For authentication, you might need to use query parameters or cookies
        if (typeof window !== 'undefined' && window.EventSource) {
            return new EventSource(url);
        } else {
            // For Node.js, you'd need a different SSE library
            throw new Error('EventSource not available in this environment');
        }
    }
    
    /**
     * Stream real-time events from the platform
     * @param {Function} callback - Function to call with each event
     * @param {Object} [filters] - Optional filters
     * @param {string} [filters.modelName] - Filter by model name
     * @param {string} [filters.evaluationStatus] - Filter by evaluation status
     * @returns {Function} Cleanup function to stop streaming
     */
    streamEvents(callback, filters = {}) {
        let url = `${this.baseUrl}/stream/events`;
        const params = new URLSearchParams();
        
        if (filters.modelName) {
            params.append('model_name', filters.modelName);
        }
        if (filters.evaluationStatus) {
            params.append('evaluation_status', filters.evaluationStatus);
        }
        
        if (params.toString()) {
            url += '?' + params.toString();
        }
        
        const eventSource = this._createEventSource(url);
        const streamId = `events_${Date.now()}`;
        
        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                callback(data);
            } catch (error) {
                console.error('Failed to parse event data:', error);
            }
        };
        
        eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
        };
        
        this.eventSources.set(streamId, eventSource);
        
        // Return cleanup function
        return () => {
            eventSource.close();
            this.eventSources.delete(streamId);
        };
    }
    
    /**
     * Stream real-time trace updates
     * @param {Function} callback - Function to call with each trace update
     * @param {number} [limit=10] - Limit for initial traces
     * @returns {Function} Cleanup function to stop streaming
     */
    streamTraces(callback, limit = 10) {
        const url = `${this.baseUrl}/stream/traces?limit=${limit}`;
        const eventSource = this._createEventSource(url);
        const streamId = `traces_${Date.now()}`;
        
        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                callback(data);
            } catch (error) {
                console.error('Failed to parse trace data:', error);
            }
        };
        
        eventSource.onerror = (error) => {
            console.error('Trace stream error:', error);
        };
        
        this.eventSources.set(streamId, eventSource);
        
        return () => {
            eventSource.close();
            this.eventSources.delete(streamId);
        };
    }
    
    /**
     * Stream real-time metrics updates
     * @param {Function} callback - Function to call with each metrics update
     * @param {number} [interval=5] - Update interval in seconds
     * @returns {Function} Cleanup function to stop streaming
     */
    streamMetrics(callback, interval = 5) {
        const url = `${this.baseUrl}/stream/metrics?interval=${interval}`;
        const eventSource = this._createEventSource(url);
        const streamId = `metrics_${Date.now()}`;
        
        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                callback(data);
            } catch (error) {
                console.error('Failed to parse metrics data:', error);
            }
        };
        
        eventSource.onerror = (error) => {
            console.error('Metrics stream error:', error);
        };
        
        this.eventSources.set(streamId, eventSource);
        
        return () => {
            eventSource.close();
            this.eventSources.delete(streamId);
        };
    }
    
    /**
     * Close all active streams
     */
    closeAllStreams() {
        for (const eventSource of this.eventSources.values()) {
            eventSource.close();
        }
        this.eventSources.clear();
    }
}

/**
 * Utility class for automatic trace logging with timing
 */
class TraceLogger {
    /**
     * Initialize trace logger
     * @param {LLMEvalClient} client - LLM Eval client instance
     * @param {string} modelName - Model name for traces
     * @param {string} [traceId] - Optional trace ID
     */
    constructor(client, modelName, traceId = null) {
        this.client = client;
        this.modelName = modelName;
        this.traceId = traceId || `trace_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.startTime = null;
        this.metadata = {};
    }
    
    /**
     * Start timing
     */
    start() {
        this.startTime = Date.now();
        return this;
    }
    
    /**
     * Log trace with timing information
     * @param {string} userQuery - User's query
     * @param {string} aiResponse - AI's response
     * @param {Object} [additionalData] - Additional trace data
     * @returns {Promise<Object>} Response from sending trace
     */
    async logTrace(userQuery, aiResponse, additionalData = {}) {
        const responseTime = this.startTime ? Date.now() - this.startTime : null;
        
        const traceData = {
            traceId: this.traceId,
            modelName: this.modelName,
            userQuery: userQuery,
            aiResponse: aiResponse,
            responseTimeMs: responseTime,
            metadata: { ...this.metadata, ...additionalData.metadata },
            ...additionalData
        };
        
        try {
            return await this.client.sendTrace(traceData);
        } catch (error) {
            console.error('Failed to log trace:', error);
            throw error;
        }
    }
    
    /**
     * Add metadata to the trace
     * @param {Object} metadata - Metadata to add
     */
    addMetadata(metadata) {
        this.metadata = { ...this.metadata, ...metadata };
        return this;
    }
}

// Example usage functions
function exampleUsage() {
    const client = new LLMEvalClient('http://localhost:8000');
    
    // Send a simple trace
    client.sendTrace({
        traceId: 'example_trace_1',
        modelName: 'gpt-4',
        userQuery: 'What is the capital of France?',
        aiResponse: 'The capital of France is Paris.',
        tokensUsed: 25,
        responseTimeMs: 850,
        cost: 0.002,
        metadata: { temperature: 0.7, maxTokens: 100 }
    }).then(response => {
        console.log('Trace sent:', response);
    }).catch(error => {
        console.error('Failed to send trace:', error);
    });
    
    // Using TraceLogger for automatic timing
    const logger = new TraceLogger(client, 'gpt-3.5-turbo').start();
    
    // Simulate some async work
    setTimeout(() => {
        logger.logTrace(
            'Explain quantum computing',
            'Quantum computing uses quantum mechanics...',
            { tokensUsed: 150, cost: 0.001 }
        );
    }, 500);
    
    // Get traces
    client.getTraces(10, 'pending').then(traces => {
        console.log(`Found ${traces.length} pending traces`);
    });
    
    // Stream real-time updates
    const streamClient = new LLMEvalStreamClient('http://localhost:8000');
    
    const stopStreaming = streamClient.streamEvents((event) => {
        console.log('Received event:', event);
    });
    
    // Stop streaming after 30 seconds
    setTimeout(() => {
        stopStreaming();
    }, 30000);
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    // CommonJS (Node.js)
    module.exports = {
        LLMEvalClient,
        LLMEvalStreamClient,
        TraceLogger
    };
} else if (typeof window !== 'undefined') {
    // Browser global
    window.LLMEvalSDK = {
        LLMEvalClient,
        LLMEvalStreamClient,
        TraceLogger
    };
}

// ES6 modules (if supported)
// export { LLMEvalClient, LLMEvalStreamClient, TraceLogger }; 