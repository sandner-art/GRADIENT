<!-- frontend/src/lib/components/shared/QuantizationSelect.svelte -->
<!-- Precision a model is loaded with (request field `quantization`); unavailable options are
     shown with the reason (backend not installed, needs an NVIDIA GPU). -->
<script lang="ts" module>
	import { get } from '#lib/apiClient.js';
	import type { QuantOption } from '#lib/types.js';

	let cached: Promise<QuantOption[]> | null = null;
	/** The options, fetched once per page (refresh: pass true). */
	export function quantOptions(refresh = false): Promise<QuantOption[]> {
		if (!cached || refresh)
			cached = get<{ options: QuantOption[] }>('/system/quantization')
				.then((r) => r.options)
				.catch(() => {
					cached = null;
					return [];
				});
		return cached;
	}
</script>

<script lang="ts">
	import { onMount } from 'svelte';
	import * as ui from '#lib/lab/ui.js';

	interface Props {
		id: string;
		value: string | null | undefined;
		onchange: (value: string | null) => void;
		label?: string;
		/** let the model decide (empty value) */
		allowDefault?: boolean;
	}
	let { id, value, onchange, label = 'Precision', allowDefault = true }: Props = $props();

	let options = $state<QuantOption[]>([]);
	onMount(async () => {
		options = await quantOptions();
	});

	const base = $derived((value ?? '').replace(/\+text$/, ''));
	const text = $derived((value ?? '').endsWith('+text'));
	const current = $derived(options.find((o) => o.value === base));

	function set(next: string, withText: boolean) {
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
				>{o.label}{o.available
					? ''
					: ` (${o.reason?.includes('GPU') ? 'needs GPU' : 'not installed'})`}</option
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
</div>
