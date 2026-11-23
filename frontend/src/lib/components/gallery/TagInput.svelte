<!-- frontend/src/lib/components/gallery/TagInput.svelte -->
<!-- Your own tags as removable chips; Enter or comma adds, Backspace on empty removes the last. -->
<script lang="ts">
	interface Props {
		tags: string[];
		suggestions?: string[];
		onchange: (tags: string[]) => void;
		id?: string;
	}
	let { tags, suggestions = [], onchange, id = 'tag-input' }: Props = $props();
	let text = $state('');

	function add(raw: string) {
		const next = raw
			.split(',')
			.map((t) => t.trim().replace(/\s+/g, ' '))
			.filter(Boolean)
			.filter((t) => !tags.includes(t));
		if (next.length) onchange([...tags, ...next]);
		text = '';
	}
</script>

<div
	class="flex flex-wrap items-center gap-1 rounded-md border border-gray-300 px-2 py-1 dark:border-gray-600 dark:bg-gray-700"
>
	{#each tags as tag (tag)}
		<span
			class="inline-flex items-center rounded bg-indigo-100 px-1.5 py-0.5 text-xs text-indigo-900 dark:bg-indigo-900 dark:text-indigo-100"
		>
			{tag}
			<button
				type="button"
				class="ml-1 opacity-60 hover:opacity-100"
				aria-label="Remove tag {tag}"
				onclick={() => onchange(tags.filter((t) => t !== tag))}>✕</button
			>
		</span>
	{/each}
	<input
		{id}
		class="min-w-24 flex-1 bg-transparent py-0.5 text-sm outline-none"
		placeholder={tags.length ? '' : 'add tags…'}
		list="{id}-suggestions"
		bind:value={text}
		onkeydown={(e) => {
			if (e.key === 'Enter' || e.key === ',') {
				e.preventDefault();
				add(text);
			} else if (e.key === 'Backspace' && !text && tags.length) {
				onchange(tags.slice(0, -1));
			}
			e.stopPropagation();
		}}
		onblur={() => text && add(text)}
	/>
	<datalist id="{id}-suggestions">
		{#each suggestions.filter((s) => !tags.includes(s)).slice(0, 50) as s (s)}<option value={s}
			></option>{/each}
	</datalist>
</div>
