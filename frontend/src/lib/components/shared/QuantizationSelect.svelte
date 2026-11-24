<!-- frontend/src/lib/components/shared/QuantizationSelect.svelte -->
<!-- Precision a model is loaded with (request field `quantization`). Options that can't be
     used for the selected model are shown greyed with the reason (backend not installed,
     needs an NVIDIA GPU, a GGUF made for another model, needs a model folder); a precision
     that doesn't fit a newly selected model is reset to the model default, with a note. -->
<script lang="ts" module>
	import { get } from '#lib/apiClient.js';
	import type { QuantOption } from '#lib/types.js';

	// a request cache, not UI state
	// eslint-disable-next-line svelte/prefer-svelte-reactivity
	const cache = new Map<string, Promise<QuantOption[]>>();
	/** The options (for a model, when given), fetched once per model (refresh: pass true). */
	export function quantOptions(
		refresh = false,
		modelId: string | null = null
	): Promise<QuantOption[]> {
		const key = modelId ?? '';
		if (refresh || !cache.has(key)) {
			const query = modelId ? `?model_id=${encodeURIComponent(modelId)}` : '';
			cache.set(
				key,
				get<{ options: QuantOption[] }>(`/system/quantization${query}`)
					.then((r) => r.options)
					.catch(() => {
						cache.delete(key);
						return [];
					})
			);
		}
		return cache.get(key)!;
	}
</script>

<script lang="ts">
	import * as ui from '#lib/lab/ui.js';

	interface Props {
		id: string;
		value: string | null | undefined;
		onchange: (value: string | null) => void;
		/** the selected model: options are checked against it */
		modelId?: string | null;
		label?: string;
		/** let the model decide (empty value) */
		allowDefault?: boolean;
	}
	let {
		id,
		value,
		onchange,
		modelId = null,
		label = 'Precision',
		allowDefault = true
	}: Props = $props();

	let options = $state<QuantOption[]>([]);
	let resetNote = $state<string | null>(null);

	const base = $derived((value ?? '').replace(/\+text$/, ''));
	const text = $derived((value ?? '').endsWith('+text'));
	const current = $derived(options.find((o) => o.value === base));

	// (re)load for the selected model; a chosen precision that doesn't fit it is reset
	$effect(() => {
		const model = modelId;
		const wanted = base; // also re-check when the value is set from outside (Use settings)
		let stale = false;
		quantOptions(false, model).then((opts) => {
			if (stale) return;
			options = opts;
			const chosen = opts.find((o) => o.value === wanted);
			if (wanted && chosen && !chosen.available && allowDefault) {
				resetNote = `${chosen.label} can't be used for ${model ?? 'this model'} (${chosen.reason}); back to the model default.`;
				onchange(null);
			}
		});
		return () => {
			stale = true;
		};
	});

	function set(next: string, withText: boolean) {
		resetNote = null;
		if (!next) return onchange(null);
		onchange(withText && next !== 'none' ? `${next}+text` : next);
	}
</script>

<div>
	<label for={id} class={ui.label}>{label}</label>
	<select {id} class={ui.input} value={base} onchange={(e) => set(e.currentTarget.value, text)}>
		{#if allowDefault}<option value="">Model default</option>{/if}
		{#each options as o (o.value)}
			<option value={o.value} disabled={!o.available} title={o.reason ?? o.about}
				>{o.label}{o.available ? '' : ` (${o.short ?? 'not available'})`}</option
			>
		{/each}
	</select>
	{#if base && base !== 'none'}
		<label class="mt-1 flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
			<input type="checkbox" checked={text} onchange={(e) => set(base, e.currentTarget.checked)} />
			Text encoders too
		</label>
		{#if current?.about}<p class={ui.help}>{current.about}</p>{/if}
	{/if}
	{#if resetNote}<p class="mt-1 text-xs text-amber-700 dark:text-amber-400" role="status">
			{resetNote}
		</p>{/if}
</div>
