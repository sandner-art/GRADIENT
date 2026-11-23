<!-- frontend/src/lib/components/Toasts.svelte -->
<script lang="ts">
	import { dismiss, ui } from '#lib/stores/ui.svelte.js';
</script>

<!-- top right; over the image (bottom left) while the viewer's details panel is open -->
<div
	class="pointer-events-none fixed z-[60] flex w-80 flex-col gap-2"
	class:top-20={!ui.viewer}
	class:right-4={!ui.viewer}
	class:bottom-4={!!ui.viewer}
	class:left-4={!!ui.viewer}
	aria-live="polite"
>
	{#each ui.toasts as toast (toast.id)}
		<div
			class="pointer-events-auto flex items-start gap-3 rounded-lg border p-3 text-sm shadow-lg"
			class:border-green-300={toast.kind === 'success'}
			class:bg-green-50={toast.kind === 'success'}
			class:text-green-900={toast.kind === 'success'}
			class:border-red-300={toast.kind === 'error'}
			class:bg-red-50={toast.kind === 'error'}
			class:text-red-900={toast.kind === 'error'}
			class:border-gray-200={toast.kind === 'info'}
			class:bg-white={toast.kind === 'info'}
			class:text-gray-900={toast.kind === 'info'}
			role={toast.kind === 'error' ? 'alert' : 'status'}
		>
			<span class="flex-1 break-words">{toast.text}</span>
			{#if toast.action}
				<button
					type="button"
					class="font-medium underline"
					onclick={() => {
						toast.action?.run();
						dismiss(toast.id);
					}}>{toast.action.label}</button
				>
			{/if}
			<button
				type="button"
				class="opacity-50 hover:opacity-100"
				aria-label="Dismiss"
				onclick={() => dismiss(toast.id)}>✕</button
			>
		</div>
	{/each}
</div>
