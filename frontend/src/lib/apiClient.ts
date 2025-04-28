// frontend/src/lib/apiClient.ts

/**
 * Base URL for the API.
 * During development, Vite proxies /api requests, so we use a relative path.
 * In production, this might point to the actual backend URL if served separately.
 */
const API_BASE_URL = '/api/v1'; // Use the proxied path configured in vite.config.ts

/**
 * Represents a generic API error structure.
 * Matching potential FastAPI/Pydantic error responses.
 */
export interface ApiError {
	status: number; // HTTP status code
	statusText: string; // HTTP status text
	detail?: any; // Pydantic validation errors list or FastAPI HTTPException detail string/object
}

/**
 * Performs a fetch request to the backend API.
 * Handles JSON parsing and structures error handling.
 * @param endpoint The API endpoint path relative to API_BASE_URL (e.g., '/generate/text-to-image').
 * @param options Standard fetch RequestInit options (method, headers, body, etc.).
 * @returns Promise resolving to the parsed JSON response <T> or rejecting with ApiError.
 */
export async function apiClient<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
	const url = `${API_BASE_URL}${endpoint}`;
	const defaultHeaders = {
		// Default headers for JSON communication
		'Content-Type': 'application/json',
		'Accept': 'application/json',
		// Allow overriding or adding headers via options
		...options.headers
	};

	console.debug(`API Request: ${options.method || 'GET'} ${url}`, options.body ? 'with body' : ''); // Basic request logging

	try {
		const response = await fetch(url, {
			...options,
			headers: defaultHeaders
		});

		// Check if the response was successful (status code 200-299)
		if (!response.ok) {
			let errorDetail: any = null;
			const contentType = response.headers.get('content-type');
			try {
                // Try to parse error body as JSON only if Content-Type indicates it
				if (contentType && contentType.includes('application/json')) {
                    errorDetail = await response.json();
                } else {
                    // Otherwise, try to get text for non-JSON error messages
                    errorDetail = await response.text();
                }
			} catch (e) {
                // If parsing fails for some reason, use status text
                console.warn('Could not parse error response body:', e)
				errorDetail = response.statusText;
			}

            // Construct a consistent error object
			const error: ApiError = {
				status: response.status,
				statusText: response.statusText,
                // Use parsed detail if available, prioritizing FastAPI's 'detail' field if object, else use raw detail/text
				detail: (typeof errorDetail === 'object' && errorDetail?.detail) ? errorDetail.detail : errorDetail
			};
			console.error(`API Error Response (${url}):`, error);
			throw error; // Throw custom error object
		}

		// Handle successful responses with no content (e.g., 204 for DELETE)
		if (response.status === 204 || response.headers.get('content-length') === '0') {
            // Return null or undefined, depending on what the calling code expects for type T
            // Returning null is often safer than undefined for JSON-based APIs.
            return null as T;
        }

		// Attempt to parse successful JSON response
		// Ensure Content-Type is JSON before parsing
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            const data: T = await response.json();
		    return data;
        } else {
            console.warn(`Expected JSON response from ${url} but received Content-Type: ${contentType}`);
            // What should we return here? Maybe text? For now, throw error.
            throw {
                status: response.status,
                statusText: "Invalid Content-Type",
                detail: `Expected JSON response but received ${contentType}`
            } as ApiError;
        }

	} catch (error) {
		// Handle network errors or errors thrown from response processing
		if (isApiError(error)) {
			// Re-throw the structured ApiError
			throw error;
		} else if (error instanceof Error) {
            // Network error or other unexpected JS error
			console.error('Network or other error:', error);
			const networkError: ApiError = {
				status: -1, // Use a specific code for network/client-side errors
				statusText: 'Network error or request failed',
				detail: error.message
			};
			throw networkError;
        } else {
            // Catch anything else that might have been thrown
            console.error('Unknown error during API call:', error);
            const unknownError: ApiError = {
                status: -2,
                statusText: 'Unknown client-side error',
                detail: String(error)
            };
            throw unknownError;
        }
	}
}

/**
 * Type guard to check if an error object conforms to the ApiError interface.
 */
export function isApiError(error: any): error is ApiError {
	return typeof error === 'object' && error !== null && typeof error.status === 'number' && typeof error.statusText === 'string';
}

// --- Convenience Methods ---

/** Makes a GET request */
export function get<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
	return apiClient<T>(endpoint, { ...options, method: 'GET' });
}

/** Makes a POST request with a JSON body */
export function post<T>(endpoint: string, body: any, options: RequestInit = {}): Promise<T> {
    // Ensure body is stringified JSON
	return apiClient<T>(endpoint, { ...options, method: 'POST', body: JSON.stringify(body) });
}

/** Makes a PUT request with a JSON body */
export function put<T>(endpoint: string, body: any, options: RequestInit = {}): Promise<T> {
    // Ensure body is stringified JSON
	return apiClient<T>(endpoint, { ...options, method: 'PUT', body: JSON.stringify(body) });
}

/** Makes a DELETE request */
export function del<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
	return apiClient<T>(endpoint, { ...options, method: 'DELETE' });
}