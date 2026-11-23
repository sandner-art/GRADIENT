<!-- frontend/src/lib/components/gallery/RatingStars.svelte -->
<script lang="ts">
	interface Props {
		value: number | null;
		onchange?: (rating: number) => void;
		size?: 'sm' | 'md';
		readonly?: boolean;
	}
	let { value, onchange, size = 'md', readonly = false }: Props = $props();
	let hover = $state(0);
	const shown = $derived(hover || value || 0);
</script>

<span
	class="inline-flex items-center"
	class:text-lg={size === 'md'}
	class:text-sm={size === 'sm'}
	role="group"
	aria-label="Rating {value ?? 0} of 5"
	onmouseleave={() => (hover = 0)}
>
	{#each [1, 2, 3, 4, 5] as n (n)}
		<button
			type="button"
			class="leading-none disabled:cursor-default"
			class:text-amber-400={n <= shown}
			class:text-gray-300={n > shown}
			class:dark:text-gray-600={n > shown}
			disabled={readonly}
			title={readonly ? undefined : n === value ? 'Clear rating' : `${n} star${n > 1 ? 's' : ''}`}
			aria-label="{n} star{n > 1 ? 's' : ''}"
			onmouseenter={() => !readonly && (hover = n)}
			onclick={(e) => {
				e.stopPropagation();
				onchange?.(n === value ? 0 : n);
			}}>★</button
		>
	{/each}
</span>
