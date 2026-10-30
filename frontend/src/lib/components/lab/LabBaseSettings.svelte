<!-- frontend/src/lib/components/lab/LabBaseSettings.svelte -->
<!-- Model, prompt and sampling settings shared by all Lab tools. -->
<script lang="ts">
	import { onMount } from 'svelte';
	import { get } from '#lib/apiClient.js';
	import RecommendedSettings from '#lib/components/shared/RecommendedSettings.svelte';
	import { lab } from '#lib/lab/labState.svelte.js';
	import * as ui from '#lib/lab/ui.js';
	import { plainSchedulerName } from '#lib/utils/format.js';

	let models = $state<string[]>([]);
	let schedulers = $state<string[]>([]);
	let loras = $state<string[]>([]);
	let loadError = $state<string | null>(null);

	const b = lab.base;

	onMount(async () => {
		const [m, s, l] = await Promise.all([
			get<string[]>('/system/models/diffusion').catch(() => null),
			get<string[]>('/system/schedulers').catch(() => ['Default']),
			get<string[]>('/system/models/loras').catch(() => [])
		]);
		if (m === null) loadError = 'Could not load the model list. Is the API running?';
		models = m ?? [];
		schedulers = s.filter((x) => x !== 'Default');
		loras = l;
		if (!b.model_id && models.length) b.model_id = models[0];
	});

	function randomSeed() {
		b.seed = Math.floor(Math.random() * 2 ** 32);
	}
</script>

<div class={ui.card}>
	<h2 class="{ui.heading} mb-3">Base settings</h2>
	<div class="space-y-3">
		<div>
			<label for="lab_model" class={ui.label}>Model</label>
			<select id="lab_model" class={ui.input} bind:value={b.model_id} disabled={!!lab.busy}>
				{#if !models.length}
					<option value="">No models installed</option>
				{/if}
				{#each models as m (m)}
					<option value={m}>{m}</option>
				{/each}
			</select>
			{#if loadError}<p class="mt-1 text-xs text-red-600">{loadError}</p>{/if}
			<RecommendedSettings
				modelId={b.model_id}
				onapply={(d) => {
					if (d.width) b.width = d.width;
					if (d.height) b.height = d.height;
					if (d.steps) b.num_inference_steps = d.steps;
					if (d.guidance_scale != null) b.guidance_scale = d.guidance_scale;
					b.scheduler_id = plainSchedulerName(d.scheduler);
				}}
			/>
		</div>

		<div>
			<label for="lab_prompt" class={ui.label}>Prompt</label>
			<textarea id="lab_prompt" rows="3" class={ui.input} bind:value={b.prompt}></textarea>
		</div>
		<div>
			<label for="lab_negative" class={ui.label}>Negative prompt</label>
			<textarea id="lab_negative" rows="2" class={ui.input} bind:value={b.negative_prompt}
			></textarea>
		</div>

		<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
			<div>
				<label for="lab_w" class={ui.label}>Width</label>
				<input id="lab_w" type="number" step="8" min="64" class={ui.input} bind:value={b.width} />
			</div>
			<div>
				<label for="lab_h" class={ui.label}>Height</label>
				<input id="lab_h" type="number" step="8" min="64" class={ui.input} bind:value={b.height} />
			</div>
			<div>
				<label for="lab_steps" class={ui.label}>Steps</label>
				<input
					id="lab_steps"
					type="number"
					min="1"
					max="150"
					class={ui.input}
					bind:value={b.num_inference_steps}
				/>
			</div>
			<div>
				<label for="lab_cfg" class={ui.label}>Guidance</label>
				<input
					id="lab_cfg"
					type="number"
					step="0.5"
					min="0"
					class={ui.input}
					bind:value={b.guidance_scale}
				/>
			</div>
		</div>

		<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
			<div>
				<label for="lab_seed" class={ui.label}>Seed</label>
				<div class="flex gap-2">
					<input
						id="lab_seed"
						type="number"
						min="0"
						placeholder="random"
						class={ui.input}
						value={b.seed ?? ''}
						oninput={(e) => {
							const v = e.currentTarget.value;
							b.seed = v === '' ? null : Number(v);
						}}
					/>
					<button
						type="button"
						class={ui.secondaryButton}
						title="New random seed"
						onclick={randomSeed}>🎲</button
					>
				</div>
				<p class={ui.help}>Keep a fixed seed so results are comparable between tools.</p>
			</div>
			<div>
				<label for="lab_scheduler" class={ui.label}>Scheduler</label>
				<select id="lab_scheduler" class={ui.input} bind:value={b.scheduler_id}>
					<option value={null}>Model default</option>
					{#each schedulers as s (s)}
						<option value={s}>{s}</option>
					{/each}
				</select>
			</div>
		</div>

		{#if loras.length}
			<div class="grid grid-cols-3 gap-3">
				<div class="col-span-2">
					<label for="lab_lora" class={ui.label}>LoRA (optional)</label>
					<select id="lab_lora" class={ui.input} bind:value={b.lora}>
						<option value={null}>None</option>
						{#each loras as l (l)}
							<option value={l}>{l}</option>
						{/each}
					</select>
				</div>
				<div>
					<label for="lab_lora_w" class={ui.label}>Weight</label>
					<input
						id="lab_lora_w"
						type="number"
						step="0.1"
						class={ui.input}
						bind:value={b.lora_weight}
						disabled={!b.lora}
					/>
				</div>
			</div>
		{/if}
	</div>
</div>
