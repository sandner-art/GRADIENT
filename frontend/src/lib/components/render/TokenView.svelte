<!-- frontend/src/lib/components/render/TokenView.svelte -->
<!-- The tokens an SD/SDXL model reads: count, 75-token chunks, weights, embeddings, and
     what plain text cuts off. Instant: the server only runs the model's tokenizer. -->
<script lang="ts">
	import { post } from '#lib/apiClient.js';
	import { tokenLine, weightStyle, type TokenView } from '#lib/tokens.js';

	interface Props {
		modelId: string | null | undefined;
		prompt: string | null | undefined;
		negative: string | null | undefined;
		syntax: string | null | undefined;
		styled?: boolean; // the prompts include styles (as rendered)
	}
	let { modelId, prompt, negative, syntax, styled = false }: Props = $props();

	let view = $state<{ prompt: TokenView | null; negative_prompt: TokenView | null } | null>(null);
	let open = $state(false);

	let timer: ReturnType<typeof setTimeout> | undefined;
	let asked = 0;
	$effect(() => {
		const body = {
			model_id: modelId ?? '',
			prompt: prompt ?? '',
			negative_prompt: negative ?? '',
			prompt_syntax: syntax ?? 'a1111'
		};
		clearTimeout(timer);
		if (!body.model_id) {
			view = null;
			return;
		}
		const ticket = ++asked;
		timer = setTimeout(async () => {
			try {
				const r = await post<{ prompt: TokenView | null; negative_prompt: TokenView | null }>(
					'/prompt/tokens',
					body
				);
				if (ticket === asked) view = r;
			} catch {
				if (ticket === asked) view = null;
			}
		}, 300);
		return () => clearTimeout(timer);
	});

	const plain = $derived((syntax ?? 'a1111') === 'plain');
</script>

{#snippet tokens(label: string, v: TokenView)}
	<div class="mt-1">
		<span class="font-medium">{label}</span>
		{#each v.chunks as chunk, c (c)}
			<div class="mt-0.5 flex flex-wrap items-baseline gap-y-0.5 leading-5">
				{#if v.chunks.length > 1}<span class="mr-1 text-[10px] text-gray-400">{c + 1}</span>{/if}
				{#each chunk as p, i (i)}
					<span
						class="rounded px-0.5 {p.emb
							? 'bg-amber-100 text-amber-900 dark:bg-amber-900 dark:text-amber-100'
							: ''} {p.end ? 'mr-1' : ''}"
						style={weightStyle(p.w)}
						title={`${p.emb ? 'embedding vector' : 'token'}${p.w !== 1 ? ` · weight ${p.w}` : ''}`}
						>{p.t}{#if p.w !== 1}<sup class="ml-px text-[9px] opacity-70">{p.w}</sup>{/if}</span
					>
				{/each}
			</div>
		{/each}
		{#if v.dropped.length}
			<div
				class="mt-0.5 flex flex-wrap gap-y-0.5 leading-5 text-red-700 line-through opacity-70 dark:text-red-300"
				title="Plain text: the model never sees these tokens"
			>
				{#each v.dropped as p, i (i)}<span class={p.end ? 'mr-1' : ''}>{p.t}</span>{/each}
			</div>
		{/if}
	</div>
{/snippet}

{#if view?.prompt}
	<div class="text-xs text-gray-500 dark:text-gray-400">
		<div class="flex flex-wrap items-center gap-x-3">
			<span
				>Tokens{styled ? ' (with styles)' : ''}: {tokenLine(
					view.prompt,
					plain
				)}{#if view.negative_prompt?.tokens}{` · negative ${tokenLine(view.negative_prompt, plain)}`}{/if}</span
			>
			<button
				type="button"
				class="text-indigo-600 hover:underline dark:text-indigo-400"
				aria-expanded={open}
				onclick={() => (open = !open)}>{open ? 'Hide' : 'Show'} tokens</button
			>
		</div>
		{#if open}
			<div
				class="mt-1 max-h-64 overflow-y-auto rounded border border-gray-200 bg-gray-50 p-2 text-gray-800 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-200"
			>
				{@render tokens('Prompt', view.prompt)}
				{#if view.negative_prompt?.tokens}{@render tokens('Negative', view.negative_prompt)}{/if}
				{#if view.prompt.edited}
					<p class="mt-1 text-gray-500">Prompt editing is shown as the prompt reads at the end.</p>
				{/if}
			</div>
		{/if}
	</div>
{/if}
