<!-- frontend/src/lib/components/queue/SweepImportDialog.svelte -->
<!-- Import a test sweep file from ArtAgents' LoRA Lab: check the model and LoRAs, then queue it. -->
<script lang="ts">
	import * as ui from '#lib/lab/ui.js';
	import { loraState, previewSweep, queueSweep } from '#lib/sweeps.js';
	import { notify } from '#lib/stores/ui.svelte.js';
	import type { SweepPreview, SweepQueued } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	interface Props {
		onclose: () => void;
		onqueued?: (result: SweepQueued) => void;
	}
	let { onclose, onqueued }: Props = $props();

	let text = $state('');
	let fileName = $state('');
	let info = $state<SweepPreview | null>(null);
	let picks = $state<Record<string, string>>({});
	let busy = $state(false);
	let error = $state<string | null>(null);
	let queued = $state<SweepQueued | null>(null);

	const lora_map = () =>
		Object.fromEntries(Object.entries(picks).filter(([, v]) => v.trim().length > 0));

	async function check() {
		if (!text) return;
		busy = true;
		error = null;
		try {
			info = await previewSweep(text, lora_map());
		} catch (e) {
			info = null;
			error = apiErrorMessage(e);
		} finally {
			busy = false;
		}
	}

	async function pickFile(e: Event) {
		const file = (e.currentTarget as HTMLInputElement).files?.[0];
		if (!file) return;
		fileName = file.name;
		text = await file.text();
		picks = {};
		await check();
	}

	async function queueAll() {
		busy = true;
		error = null;
		try {
			queued = await queueSweep(text, lora_map());
			onqueued?.(queued);
		} catch (e) {
			error = apiErrorMessage(e);
		} finally {
			busy = false;
		}
	}

	async function copyPath(path: string) {
		try {
			await navigator.clipboard.writeText(path);
			notify('Folder path copied', 'success', undefined, 1500);
		} catch {
			notify('The browser blocked clipboard access', 'error');
		}
	}
</script>

<svelte:window
	onkeydown={(e) => {
		if (e.key === 'Escape') onclose();
	}}
/>

<div class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
	<div
		class="{ui.card} max-h-full w-full max-w-2xl space-y-4 overflow-y-auto"
		role="dialog"
		aria-modal="true"
		aria-label="Import a test sweep"
	>
		<div class="flex items-start justify-between">
			<div>
				<h2 class={ui.heading}>Import a test sweep</h2>
				<p class={ui.muted}>
					A <code>.lora-sweep.json</code> file from ArtAgents' LoRA Lab (Test Sweep → Export sweep for
					GRADIENT).
				</p>
			</div>
			<button
				type="button"
				class="text-2xl leading-none text-gray-500"
				aria-label="Close"
				onclick={onclose}>×</button
			>
		</div>

		{#if queued}
			<div class="space-y-2 text-sm">
				<p>
					Queued <strong>{queued.queued}</strong> image{queued.queued === 1 ? '' : 's'} of
					<strong>{queued.name}</strong>. They show in the gallery as one run (<em>Group</em>).
				</p>
				<p>When the last one is done, the results folder for ArtAgents is written here:</p>
				<div class="flex items-center gap-2">
					<code
						class="min-w-0 flex-1 truncate rounded bg-gray-100 px-2 py-1 text-xs dark:bg-gray-900"
						data-testid="sweep-results-dir">{queued.results_dir}</code
					>
					<button
						type="button"
						class={ui.secondaryButton}
						onclick={() => copyPath(queued!.results_dir)}>Copy path</button
					>
				</div>
				<p class={ui.help}>
					In ArtAgents: LoRA Lab → Test Sweep → <em>Import GRADIENT results</em> with this folder.
				</p>
				<div class="flex justify-end">
					<button type="button" class={ui.primaryButton} onclick={onclose}>Done</button>
				</div>
			</div>
		{:else}
			<div>
				<label class={ui.label} for="sweep_file">Sweep file</label>
				<input
					id="sweep_file"
					type="file"
					accept=".json,application/json"
					class="block w-full text-sm"
					onchange={pickFile}
				/>
				{#if fileName}<p class={ui.help}>{fileName}</p>{/if}
			</div>

			{#if error}<div class={ui.errorBox}>{error}</div>{/if}

			{#if info}
				<div class="space-y-1 text-sm">
					<p><strong>{info.name}</strong></p>
					<p>
						{info.images} image{info.images === 1 ? '' : 's'} · {info.prompts} prompt{info.prompts ===
						1
							? ''
							: 's'} · {info.conditions} condition{info.conditions === 1 ? '' : 's'}
					</p>
					<p>
						Model <code>{info.model_id}</code>
						{#if info.model_found}<span class="text-green-600">✓</span>
						{:else}<span class="text-red-600">✗ not installed here</span>{/if}
					</p>
					{#if info.model_defaults.steps || info.model_defaults.guidance_scale !== undefined}
						<p class={ui.help}>
							Steps or CFG the sweep leaves open use the model's recommended
							{#if info.model_defaults.steps}{info.model_defaults.steps} steps{/if}
							{#if info.model_defaults.guidance_scale !== undefined}CFG {info.model_defaults
									.guidance_scale}{/if}.
						</p>
					{/if}
					{#if Object.keys(info.options).length}
						<p class={ui.help}>
							{Object.entries(info.options)
								.map(([k, v]) => `${k}: ${v}`)
								.join(' · ')}
						</p>
					{/if}
				</div>

				{#if info.loras.length}
					<table class="w-full text-left text-sm">
						<thead class="text-xs text-gray-500">
							<tr><th class="py-1">LoRA</th><th>Here</th></tr>
						</thead>
						<tbody>
							{#each info.loras as lora (lora.key)}
								{@const state = loraState(lora.status)}
								<tr class="border-t border-gray-200 align-top dark:border-gray-700">
									<td class="py-1 pr-2">
										<div class="font-medium">{lora.key}</div>
										<div class="text-xs break-all text-gray-500" title={lora.sha256 ?? ''}>
											{lora.name}
										</div>
									</td>
									<td class="py-1">
										<div
											class:text-green-600={state.ok && lora.status !== 'unverified'}
											class:text-amber-600={lora.status === 'unverified'}
											class:text-red-600={!state.ok}
										>
											{state.mark}
											{lora.message}
										</div>
										{#if lora.resolved}<div class="text-xs break-all text-gray-500">
												{lora.resolved}
											</div>{/if}
										{#if !state.ok || picks[lora.key]}
											<input
												class="{ui.smallInput} mt-1 font-mono"
												placeholder="Full path to the LoRA file here"
												aria-label="Local file for {lora.key}"
												bind:value={picks[lora.key]}
											/>
										{/if}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				{/if}

				{#if info.problems.length}
					<div class={ui.noteBox}>
						{#each info.problems as problem, i (i)}<div>{problem}</div>{/each}
					</div>
				{/if}

				<div class="flex items-center justify-between gap-2">
					<p class="min-w-0 truncate text-xs text-gray-500" title={info.results_dir}>
						Results: {info.results_dir}
					</p>
					<span class="flex flex-none gap-2">
						<button type="button" class={ui.secondaryButton} disabled={busy} onclick={check}
							>Check again</button
						>
						<button
							type="button"
							class={ui.primaryButton}
							disabled={busy || !info.ready}
							onclick={queueAll}>Queue all {info.images}</button
						>
					</span>
				</div>
			{/if}
		{/if}
	</div>
</div>
