<!-- frontend/src/lib/components/LLMChat.svelte -->
<script lang="ts">
	import { get, post, isApiError, type ApiError } from '#lib/apiClient.js';
	import { onMount, tick } from 'svelte'; // Import tick for scrolling

	// --- Types (Mirroring backend models - ideally move to types.ts) ---
	interface LLMConfig {
		model_id: string;
		n_ctx?: number;
		n_gpu_layers?: number;
	}
	interface GenerationParams {
		temperature?: number;
		top_k?: number;
		top_p?: number;
		repeat_penalty?: number;
		max_tokens?: number;
		stop?: string[] | null;
		seed?: number | null;
	}
	interface ChatMessage {
		role: 'system' | 'user' | 'assistant';
		content: string;
	}
	interface ChatCompletionRequest {
		llm_config: LLMConfig;
		messages: ChatMessage[];
		generation_params?: GenerationParams;
		stream?: boolean;
	}
    interface ChatCompletionChoice {
        message: ChatMessage;
        finish_reason?: string | null;
        index?: number;
    }
	interface ChatCompletionResponse {
        id: string;
        model: string;
        choices: ChatCompletionChoice[];
        usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null;
	}
    // --- End Types ---

    // Component State
	let messages: ChatMessage[] = [];
	let currentInput: string = '';
	let isLoading: boolean = false;
	let error: string | null = null;
	let llmConfig: LLMConfig = {
		model_id: 'Hermes-3-Llama-3.2-3B.Q5_K_M.gguf', // Default GGUF model
        n_ctx: 2048,
        n_gpu_layers: 0 // Default to CPU initially
	};
    let generationParams: GenerationParams = {
        temperature: 0.7,
        max_tokens: 512,
        stop: null // No default stop tokens
    };
    let useStreaming: boolean = true; // Default to streaming

    let chatContainerElement: HTMLElement; // For auto-scrolling

    // GGUF files found in the llm model folder (download more in the Models tab)
    let availableLlms: string[] = [];
    onMount(async () => {
        try {
            availableLlms = await get<string[]>('/system/models/llm');
            if (availableLlms.length && !availableLlms.includes(llmConfig.model_id)) {
                llmConfig.model_id = availableLlms[0];
            }
        } catch (e) {
            console.warn('Could not list LLM models', e);
        }
    });

    // Function to add a message and scroll down
    async function addMessage(message: ChatMessage) {
        messages = [...messages, message];
        await tick(); // Wait for DOM update
        if (chatContainerElement) {
            chatContainerElement.scrollTop = chatContainerElement.scrollHeight;
        }
    }

	async function sendMessage() {
		if (!currentInput.trim() || isLoading) return;

		const userMessage: ChatMessage = { role: 'user', content: currentInput };
        addMessage(userMessage); // Add user message immediately

		const messageHistory = [...messages]; // Send the current history
        const currentInputValue = currentInput; // Store input before clearing
		currentInput = ''; // Clear input field
		isLoading = true;
		error = null;

        const payload: ChatCompletionRequest = {
            llm_config: { // Ensure types are correct
                ...llmConfig,
                n_ctx: Number(llmConfig.n_ctx) || 2048,
                n_gpu_layers: Number(llmConfig.n_gpu_layers) || 0
            },
            messages: messageHistory,
            generation_params: {
                ...generationParams,
                temperature: Number(generationParams.temperature) || 0.7,
                top_k: Number(generationParams.top_k) || 40,
                top_p: Number(generationParams.top_p) || 0.95,
                repeat_penalty: Number(generationParams.repeat_penalty) || 1.1,
                max_tokens: Number(generationParams.max_tokens) || 512,
                seed: generationParams.seed === null || isNaN(Number(generationParams.seed)) ? null : Number(generationParams.seed),
                stop: generationParams.stop?.filter(s => s.trim() !== '') ?? null // Filter empty stop strings
            },
            stream: useStreaming
        };

        console.log("Sending LLM payload:", payload);

        if (useStreaming) {
            let assistantMessage: ChatMessage = { role: 'assistant', content: '' };
            let messageAdded = false; // Flag to add assistant message container once

            try {
                 // Use fetch directly for streaming with EventSource compatibility
                 const response = await fetch('/api/v1/llm/chat/completions', {
                     method: 'POST',
                     headers: {
                         'Content-Type': 'application/json',
                         'Accept': 'text/event-stream' // Important for SSE
                     },
                     body: JSON.stringify(payload)
                 });

                 if (!response.ok || !response.body) {
                     let errorDetail = response.statusText;
                     try {
                         const errJson = await response.json();
                         errorDetail = errJson.detail || errorDetail;
                     } catch (e) { /* ignore */ }
                     throw { status: response.status, statusText: response.statusText, detail: errorDetail } as ApiError;
                 }

                 const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();

                 while (true) {
                     const { value, done } = await reader.read();
                     if (done) break;

                     // Process SSE messages (data: {...}\n\n)
                     const lines = value.split('\n');
                     for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const jsonData = line.substring(6).trim();
                            if (jsonData) {
                                try {
                                    // --- Add check for potential special end signal ---
                                    if (jsonData === "[DONE]") { // Example end signal, if backend sends one
                                        console.log("Received stream end signal.");
                                        break; // Exit inner loop if backend sends specific end marker
                                    }

                                    const chunk = JSON.parse(jsonData); // Attempt parsing

                                    if (chunk.error) { // Check for explicit errors sent by backend
                                        throw new Error(`Stream Error from backend: ${chunk.error}`);
                                    }

                                    const deltaContent = chunk?.choices?.[0]?.delta?.content;
                                    if (deltaContent !== undefined && deltaContent !== null) { // Check exists, even if empty string
                                        if (!messageAdded) {
                                            addMessage(assistantMessage);
                                            messageAdded = true;
                                        }
                                        assistantMessage.content += deltaContent;
                                        messages = messages; // Trigger reactivity
                                        await tick();
                                        if (chatContainerElement) {
                                            chatContainerElement.scrollTop = chatContainerElement.scrollHeight;
                                        }
                                    }
                                } catch (parseError: unknown) { // Catch specifically JSON parse errors
                                    // Log more informatively and SKIP this chunk
                                    console.error("Failed to parse stream chunk JSON:", parseError, "Skipped Chunk Data:", jsonData);
                                    // Optionally display a subtle indicator in the UI that a chunk was skipped?
                                    // error = "Failed to process some stream response chunks."; // Maybe too intrusive
                                }
                            }
                        }
                     }
                 }
                 console.log("Stream finished.");

            } catch (e: unknown) {
                 console.error('LLM Streaming failed:', e);
                 if (!messageAdded && assistantMessage.content === '') {
                    // If stream failed before adding anything, add user message back to input? Or show error.
                 }
                 if (isApiError(e)) {
                     error = `API Error (${e.status}): ${JSON.stringify(e.detail || e.statusText)}`;
                 } else if (e instanceof Error) {
                     error = `Stream Error: ${e.message}`;
                 } else {
                     error = 'An unknown error occurred during streaming.';
                 }
            } finally {
                 isLoading = false;
            }

        } else { // Non-streaming
            try {
                const response = await post<ChatCompletionResponse>('/llm/chat/completions', payload);
                if (response.choices && response.choices.length > 0) {
                    addMessage(response.choices[0].message); // Add assistant's full response
                } else {
                    throw new Error("Received empty response from LLM.");
                }
                console.log('LLM Non-streaming successful:', response);
            } catch (e: unknown) {
                console.error('LLM Non-streaming failed:', e);
                if (isApiError(e)) {
                    error = `API Error (${e.status}): ${JSON.stringify(e.detail || e.statusText)}`;
                } else if (e instanceof Error) {
                    error = `Error: ${e.message}`;
                } else {
                    error = 'An unknown error occurred during generation.';
                }
            } finally {
                isLoading = false;
            }
        }
	}

    function handleKeyPress(event: KeyboardEvent) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent new line on Enter
            sendMessage();
        }
    }

</script>

<div class="flex flex-col h-[70vh] max-w-3xl mx-auto bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
    <h2 class="text-xl font-semibold p-4 border-b dark:border-gray-700 text-center text-gray-800 dark:text-gray-200">LLM Chat</h2>

    <!-- Configuration Options (Collapsible?) -->
    <details class="p-4 border-b dark:border-gray-700 text-sm">
        <summary class="cursor-pointer font-medium text-gray-700 dark:text-gray-300">Configuration</summary>
        <div class="mt-4 space-y-3">
            <div>
                <label for="llm_model_id" class="block text-xs font-medium mb-1 text-gray-600 dark:text-gray-400">Model Filename:</label>
			    <input type="text" id="llm_model_id" list="llm_model_options" bind:value={llmConfig.model_id} required class="w-full px-2 py-1 border text-xs border-gray-300 rounded shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
                <datalist id="llm_model_options">
                    {#each availableLlms as name (name)}<option value={name}></option>{/each}
                </datalist>
                {#if availableLlms.length === 0}
                    <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">No GGUF models found. Download one in the Models tab (LLM filter).</p>
                {/if}
            </div>
            <div class="grid grid-cols-2 gap-3">
                 <div>
                    <label for="llm_ctx" class="block text-xs font-medium mb-1 text-gray-600 dark:text-gray-400">Context Size (n_ctx):</label>
                    <input type="number" id="llm_ctx" bind:value={llmConfig.n_ctx} min="512" step="512" class="w-full px-2 py-1 border text-xs border-gray-300 rounded shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
                </div>
                 <div>
                    <label for="llm_gpu" class="block text-xs font-medium mb-1 text-gray-600 dark:text-gray-400">GPU Layers (n_gpu):</label>
                    <input type="number" id="llm_gpu" bind:value={llmConfig.n_gpu_layers} min="-1" step="1" class="w-full px-2 py-1 border text-xs border-gray-300 rounded shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
                </div>
            </div>
             <div>
                <label for="llm_max_tokens" class="block text-xs font-medium mb-1 text-gray-600 dark:text-gray-400">Max Tokens:</label>
                <input type="number" id="llm_max_tokens" bind:value={generationParams.max_tokens} min="1" step="64" class="w-full px-2 py-1 border text-xs border-gray-300 rounded shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
            </div>
             <div class="flex items-center pt-2">
                <input type="checkbox" id="streaming" bind:checked={useStreaming} class="h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500">
                <label for="streaming" class="ml-2 block text-xs font-medium text-gray-700 dark:text-gray-300">Use Streaming</label>
            </div>
            <!-- Add sliders/inputs for temp, top_k, top_p, repeat_penalty later -->
        </div>
    </details>

    <!-- Message Display Area -->
    <div bind:this={chatContainerElement} class="flex-grow p-4 overflow-y-auto space-y-4">
        {#each messages as message (message.role + message.content.slice(0, 10) + Math.random())} <!-- Basic keying for reactivity -->
            {#if message.role === 'user'}
                <div class="flex justify-end">
                    <div class="bg-blue-500 text-white p-3 rounded-lg max-w-xs lg:max-w-md shadow">
                        {message.content}
                    </div>
                </div>
            {:else if message.role === 'assistant'}
                 <div class="flex justify-start">
                    <div class="bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-200 p-3 rounded-lg max-w-xs lg:max-w-md shadow whitespace-pre-wrap">
                        {message.content || '...'}
                    </div>
                </div>
             {:else} <!-- System message? -->
                  <div class="text-center text-xs text-gray-500 dark:text-gray-400 italic p-2">
                        System: {message.content}
                  </div>
            {/if}
        {/each}
        {#if isLoading}
             <div class="flex justify-start">
                 <div class="bg-gray-200 dark:bg-gray-600 p-3 rounded-lg shadow animate-pulse max-w-[50px]">...</div>
             </div>
        {/if}
    </div>

     <!-- Error Display -->
     {#if error}
        <div class="p-2 text-center bg-red-100 border-t border-red-300 text-red-700 dark:bg-red-900 dark:text-red-200 dark:border-red-700 text-xs" role="alert">
            <strong class="font-bold">Error:</strong> {error}
        </div>
    {/if}

    <!-- Input Area -->
    <div class="p-4 border-t dark:border-gray-700">
        <div class="flex items-center space-x-2">
            <textarea
                bind:value={currentInput}
                on:keypress={handleKeyPress}
                disabled={isLoading}
                rows="1"
                placeholder="Type your message..."
                class="flex-grow px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white resize-none"
                style="min-height: 40px; max-height: 150px;"
                on:input={(e) => { e.currentTarget.style.height = 'auto'; e.currentTarget.style.height = e.currentTarget.scrollHeight + 'px'; }}
            ></textarea>
            <button
                on:click={sendMessage}
                disabled={isLoading || !currentInput.trim()}
                class="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                Send
                 {#if isLoading}
                 <!-- Simple spinner -->
                  <svg class="animate-spin ml-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                {/if}
            </button>
        </div>
    </div>
</div>
