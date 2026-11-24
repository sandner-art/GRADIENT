<!-- frontend/src/lib/components/ContextMenu.svelte -->
<!-- The app's right-click menu (one at a time; see openMenu in stores/ui.svelte.ts). -->
<script lang="ts">
	import { tick } from 'svelte';
	import { closeMenu, ui, type MenuItem } from '#lib/stores/ui.svelte.js';

	let el = $state<HTMLDivElement | null>(null);
	let pos = $state({ x: 0, y: 0 });

	// keep the menu inside the window
	$effect(() => {
		const m = ui.menu;
		if (!m) return;
		pos = { x: m.x, y: m.y };
		tick().then(() => {
			if (!el) return;
			const r = el.getBoundingClientRect();
			pos = {
				x: Math.max(4, Math.min(m.x, window.innerWidth - r.width - 4)),
				y: Math.max(4, Math.min(m.y, window.innerHeight - r.height - 4))
			};
			el.querySelector<HTMLElement>('[role=menuitem]:not([disabled])')?.focus();
		});
	});

	async function run(item: MenuItem) {
		if (!('run' in item) || item.disabled) return;
		if (!item.keepOpen) closeMenu();
		await item.run();
	}

	function onKey(e: KeyboardEvent) {
		if (!ui.menu) return;
		if (e.key === 'Escape') {
			e.preventDefault();
			closeMenu();
			return;
		}
		if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
		e.preventDefault();
		const items = [...(el?.querySelectorAll<HTMLElement>('[role=menuitem]:not([disabled])') ?? [])];
		const i = items.indexOf(document.activeElement as HTMLElement);
		const next =
			e.key === 'ArrowDown' ? (i + 1) % items.length : (i - 1 + items.length) % items.length;
		items[next]?.focus();
	}
</script>

<svelte:window
	onkeydown={onKey}
	onresize={closeMenu}
	onblur={closeMenu}
	onpointerdown={(e) => {
		if (ui.menu && el && !el.contains(e.target as Node)) closeMenu();
	}}
	onwheel={(e) => {
		if (ui.menu && el && !el.contains(e.target as Node)) closeMenu();
	}}
/>

{#if ui.menu}
	<div
		bind:this={el}
		class="fixed z-[70] min-w-52 rounded-md border border-gray-200 bg-white py-1 text-sm shadow-xl dark:border-gray-700 dark:bg-gray-800"
		style:left="{pos.x}px"
		style:top="{pos.y}px"
		role="menu"
		tabindex="-1"
		aria-label={ui.menu.title ?? 'Actions'}
		oncontextmenu={(e) => e.preventDefault()}
	>
		{#if ui.menu.title}
			<div
				class="truncate px-3 py-1 text-xs text-gray-500 dark:text-gray-400"
				title={ui.menu.title}
			>
				{ui.menu.title}
			</div>
		{/if}
		{#each ui.menu.items as item, i (i)}
			{#if 'separator' in item}
				<div class="my-1 border-t border-gray-200 dark:border-gray-700" role="separator"></div>
			{:else if 'rating' in item}
				<div class="flex items-center gap-0.5 px-3 py-1" role="group" aria-label="Rating">
					{#each [1, 2, 3, 4, 5] as star (star)}
						<button
							type="button"
							role="menuitem"
							class="px-0.5 text-lg leading-none hover:scale-110 focus:outline-none focus-visible:ring-2"
							class:text-amber-400={(item.rating ?? 0) >= star}
							class:text-gray-300={(item.rating ?? 0) < star}
							aria-label="{star} star{star > 1 ? 's' : ''}"
							onclick={() => {
								closeMenu();
								item.rate(item.rating === star ? 0 : star);
							}}>★</button
						>
					{/each}
				</div>
			{:else}
				<button
					type="button"
					role="menuitem"
					class="block w-full px-3 py-1.5 text-left hover:bg-gray-100 focus:bg-gray-100 focus:outline-none disabled:opacity-40 dark:hover:bg-gray-700 dark:focus:bg-gray-700"
					class:text-red-600={item.danger}
					class:dark:text-red-400={item.danger}
					disabled={item.disabled}
					onclick={() => run(item)}>{item.label}</button
				>
			{/if}
		{/each}
	</div>
{/if}
