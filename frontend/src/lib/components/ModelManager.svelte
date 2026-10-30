<!-- frontend/src/lib/components/ModelManager.svelte -->
<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { del, get, isApiError, post } from '#lib/apiClient.js';
	import type {
		CatalogEntry,
		DownloadJob,
		DownloadRequest,
		LocalModel,
		ModelType,
		StorageInfo
	} from '#lib/types.js';
	import { etaSeconds, formatBytes, formatDuration } from '#lib/utils/format.js';

	const TYPE_LABELS: Record<ModelType, string> = {
		diffusers: 'Base model',
		checkpoints: 'Checkpoint',
		vae: 'VAE',
		lora: 'LoRA',
		controlnet: 'ControlNet',
		embeddings: 'Embedding',
		llm: 'LLM (GGUF)'
	};
	const FILTERS: { id: 'all' | ModelType; label: string }[] = [
		{ id: 'all', label: 'All' },
		{ id: 'diffusers', label: 'Base models' },
		{ id: 'lora', label: 'LoRA' },
		{ id: 'vae', label: 'VAE' },
		{ id: 'controlnet', label: 'ControlNet' },
		{ id: 'llm', label: 'LLM' }
	];
	const DONE = new Set(['completed', 'failed', 'cancelled']);
	const BIG_DOWNLOAD_GB = 30;

	let catalog = $state<CatalogEntry[]>([]);
	let local = $state<LocalModel[]>([]);
	let jobs = $state<DownloadJob[]>([]);
	let storage = $state<StorageInfo | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let filter = $state<'all' | ModelType>('all');
	let search = $state('');
	let sessionToken = $state('');

	// "Add your own" form
	let customRef = $state('');
	let customType = $state<ModelType | ''>('');
	let customFile = $state('');
	let submitting = $state(false);

	const activeJobs = $derived(jobs.filter((j) => !DONE.has(j.status)));
	const jobByCatalog = $derived(
		new Map(activeJobs.filter((j) => j.catalog_id).map((j) => [j.catalog_id as string, j]))
	);
	const filteredCatalog = $derived(
		catalog.filter((e) => {
			if (filter !== 'all' && e.type !== filter) return false;
			if (!search.trim()) return true;
			const hay = [e.name, e.family, e.repo_id, e.license, ...e.tags].join(' ').toLowerCase();
			return hay.includes(search.trim().toLowerCase());
		})
	);
	const localGroups = $derived(
		(Object.keys(TYPE_LABELS) as ModelType[])
			.map((t) => ({ type: t, items: local.filter((m) => m.type === t) }))
			.filter((g) => g.items.length > 0)
	);
	const freeGb = $derived(storage?.search_dirs?.[0]?.free_gb ?? null);
	const hasToken = $derived(Boolean(storage?.hf_token_configured || sessionToken.trim()));

	function describeError(e: unknown): string {
		if (isApiError(e)) {
			if (typeof e.detail === 'string') return e.detail;
			if (Array.isArray(e.detail))
				return e.detail.map((d) => d.msg ?? JSON.stringify(d)).join('; ');
			return `${e.status} ${e.statusText}`;
		}
		return e instanceof Error ? e.message : String(e);
	}

	async function refreshLists() {
		const [c, l, s] = await Promise.all([
			get<CatalogEntry[]>('/models/catalog'),
			get<LocalModel[]>('/models/local'),
			get<StorageInfo>('/models/storage')
		]);
		catalog = c;
		local = l;
		storage = s;
	}

	async function refreshJobs() {
		const before = new Set(activeJobs.map((j) => j.id));
		jobs = await get<DownloadJob[]>('/models/downloads');
		const finished = jobs.some((j) => before.has(j.id) && DONE.has(j.status));
		if (finished) await refreshLists();
	}

	// Poll quickly while something is downloading, slowly otherwise.
	let timer: ReturnType<typeof setTimeout> | undefined;
	let destroyed = false;
	async function poll() {
		try {
			await refreshJobs();
		} catch {
			/* backend restarting - keep polling */
		}
		if (!destroyed) timer = setTimeout(poll, activeJobs.length ? 1000 : 5000);
	}

	onMount(async () => {
		try {
			await Promise.all([refreshLists(), refreshJobs()]);
		} catch (e) {
			error = `Could not reach the GRADIENT API: ${describeError(e)}`;
		} finally {
			loading = false;
		}
		timer = setTimeout(poll, 1000);
	});
	onDestroy(() => {
		destroyed = true;
		clearTimeout(timer);
	});

	async function startDownload(req: DownloadRequest, label: string, sizeGb?: number | null) {
		if (sizeGb && sizeGb >= BIG_DOWNLOAD_GB) {
			const ok = confirm(
				`${label} is a ${sizeGb.toFixed(0)} GB download` +
					(freeGb !== null ? ` (${freeGb} GB free on the model disk)` : '') +
					'. Continue?'
			);
			if (!ok) return;
		}
		error = null;
		try {
			await post<DownloadJob>('/models/downloads', {
				...req,
				token: sessionToken.trim() || null
			});
			await refreshJobs();
			clearTimeout(timer);
			timer = setTimeout(poll, 1000);
		} catch (e) {
			error = describeError(e);
		}
	}

	async function submitCustom(event: SubmitEvent) {
		event.preventDefault();
		const ref = customRef.trim();
		if (!ref) return;
		submitting = true;
		const isUrl = /^https:\/\//.test(ref) && !/huggingface\.co|hf\.co/.test(ref);
		const req: DownloadRequest = isUrl ? { url: ref } : { repo_id: ref };
		if (customType) req.model_type = customType;
		if (customFile.trim()) req.files = [customFile.trim()];
		await startDownload(req, ref);
		submitting = false;
		if (!error) {
			customRef = '';
			customFile = '';
		}
	}

	async function cancelJob(job: DownloadJob) {
		try {
			await del(`/models/downloads/${job.id}`);
			await refreshJobs();
		} catch (e) {
			error = describeError(e);
		}
	}

	async function clearFinished() {
		await post('/models/downloads/clear', {});
		await refreshJobs();
	}

	async function removeLocal(model: LocalModel) {
		if (!confirm(`Delete ${model.name} (${formatBytes(model.size_bytes)}) from disk?`)) return;
		try {
			await del(`/models/local/${model.type}/${encodeURIComponent(model.name)}`);
			await refreshLists();
		} catch (e) {
			error = describeError(e);
		}
	}

	function percent(job: DownloadJob): number | null {
		return job.progress === null || job.progress === undefined
			? null
			: Math.round(job.progress * 100);
	}
</script>

<div class="space-y-6">
	<!-- Header / storage -->
	<section
		class="rounded-lg border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800"
	>
		<div class="flex flex-wrap items-baseline justify-between gap-2">
			<h2 class="text-2xl font-semibold text-gray-800 dark:text-gray-100">Model Manager</h2>
			{#if storage}
				<p class="text-sm text-gray-500 dark:text-gray-400">
					Downloads go to <code class="text-xs">{storage.download_dir}</code>
					{#if freeGb !== null}· <span class="font-medium">{freeGb} GB free</span>{/if}
				</p>
			{/if}
		</div>
		<div class="mt-3 grid gap-3 text-sm md:grid-cols-[1fr_auto] md:items-center">
			<p class="text-gray-600 dark:text-gray-300">
				Gated models (marked 🔒) need you to accept their licence on Hugging Face and a token.
				{#if storage?.hf_token_configured}
					<span class="font-medium text-emerald-700 dark:text-emerald-400"
						>A token is configured.</span
					>
				{:else}
					Set <code>HF_TOKEN</code> before starting GRADIENT, run <code>hf auth login</code>, or
					paste a token here for this session (not saved).
				{/if}
			</p>
			{#if !storage?.hf_token_configured}
				<input
					type="password"
					autocomplete="off"
					placeholder="hf_… (optional)"
					bind:value={sessionToken}
					aria-label="Hugging Face token for this session"
					class="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm md:w-64 dark:border-gray-600 dark:bg-gray-700"
				/>
			{/if}
		</div>
	</section>

	{#if error}
		<div
			class="rounded border border-red-400 bg-red-50 p-3 text-sm text-red-800 dark:border-red-700 dark:bg-red-950 dark:text-red-200"
			role="alert"
		>
			{error}
			<button class="ml-2 underline" onclick={() => (error = null)}>dismiss</button>
		</div>
	{/if}

	<!-- Downloads -->
	{#if jobs.length > 0}
		<section
			class="rounded-lg border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800"
		>
			<div class="mb-3 flex items-center justify-between">
				<h3 class="text-lg font-semibold">Downloads</h3>
				{#if jobs.some((j) => DONE.has(j.status))}
					<button
						class="text-sm text-indigo-600 hover:underline dark:text-indigo-400"
						onclick={clearFinished}>Clear finished</button
					>
				{/if}
			</div>
			<ul class="space-y-3">
				{#each jobs as job (job.id)}
					{@const pct = percent(job)}
					{@const eta = etaSeconds(job.total_bytes, job.downloaded_bytes, job.speed_bps)}
					<li class="rounded border border-gray-100 p-3 dark:border-gray-700">
						<div class="flex flex-wrap items-center justify-between gap-2">
							<div class="min-w-0">
								<p class="truncate font-medium">{job.name}</p>
								<p class="truncate text-xs text-gray-500 dark:text-gray-400">
									{job.model_type ? TYPE_LABELS[job.model_type] : ''}
									{#if job.target_path}→ {job.target_path}{/if}
								</p>
							</div>
							<div class="flex items-center gap-3 text-sm">
								{#if job.status === 'downloading' || job.status === 'resolving' || job.status === 'queued'}
									<span class="text-gray-600 tabular-nums dark:text-gray-300">
										{formatBytes(job.downloaded_bytes)} / {formatBytes(job.total_bytes)}
										{#if job.speed_bps > 0}· {formatBytes(job.speed_bps)}/s{/if}
										{#if eta !== null}· {formatDuration(eta)} left{/if}
									</span>
									<button
										class="rounded border border-gray-300 px-2 py-0.5 text-xs hover:bg-gray-100 dark:border-gray-600 dark:hover:bg-gray-700"
										onclick={() => cancelJob(job)}>Cancel</button
									>
								{:else if job.status === 'completed'}
									<span class="font-medium text-emerald-700 dark:text-emerald-400">✓ Installed</span
									>
								{:else if job.status === 'cancelled'}
									<span class="text-gray-500"
										>Cancelled (partial files kept; download again to resume)</span
									>
								{:else}
									<span class="font-medium text-red-700 dark:text-red-400">Failed</span>
								{/if}
							</div>
						</div>
						{#if !DONE.has(job.status)}
							<div
								class="mt-2 h-2 overflow-hidden rounded bg-gray-200 dark:bg-gray-700"
								role="progressbar"
								aria-label={`Download progress for ${job.name}`}
								aria-valuemin="0"
								aria-valuemax="100"
								aria-valuenow={pct ?? undefined}
							>
								<div
									class="h-full bg-indigo-500 transition-all duration-500"
									class:animate-pulse={pct === null}
									style:width={pct === null ? '30%' : `${pct}%`}
								></div>
							</div>
						{/if}
						{#if job.status === 'failed' && job.error}
							<p class="mt-2 text-sm text-red-700 dark:text-red-300">{job.error}</p>
							{#if job.error_code === 'gated'}
								<p class="mt-1 text-xs text-gray-600 dark:text-gray-400">
									1. Open <a
										class="text-indigo-600 underline dark:text-indigo-400"
										href={`https://huggingface.co/${job.source}`}
										target="_blank"
										rel="noreferrer">huggingface.co/{job.source}</a
									>
									and accept the licence. 2. Create a read token at
									<a
										class="text-indigo-600 underline dark:text-indigo-400"
										href="https://huggingface.co/settings/tokens"
										target="_blank"
										rel="noreferrer">huggingface.co/settings/tokens</a
									>. 3. Paste it above (or set HF_TOKEN) and download again.
								</p>
							{/if}
						{/if}
					</li>
				{/each}
			</ul>
		</section>
	{/if}

	<div class="grid gap-6 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
		<!-- Catalog -->
		<section
			class="rounded-lg border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800"
		>
			<div class="mb-4 flex flex-wrap items-center justify-between gap-3">
				<h3 class="text-lg font-semibold">Catalog</h3>
				<input
					type="search"
					placeholder="Search (flux, fast, apache, sdxl…)"
					bind:value={search}
					class="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm sm:w-72 dark:border-gray-600 dark:bg-gray-700"
				/>
			</div>
			<div class="mb-4 flex flex-wrap gap-2" role="group" aria-label="Filter by model type">
				{#each FILTERS as f (f.id)}
					<button
						class="rounded-full border px-3 py-1 text-sm"
						class:bg-indigo-600={filter === f.id}
						class:text-white={filter === f.id}
						class:border-indigo-600={filter === f.id}
						class:border-gray-300={filter !== f.id}
						class:dark:border-gray-600={filter !== f.id}
						aria-pressed={filter === f.id}
						onclick={() => (filter = f.id)}>{f.label}</button
					>
				{/each}
			</div>

			{#if loading}
				<p class="text-sm text-gray-500">Loading catalog…</p>
			{:else if filteredCatalog.length === 0}
				<p class="text-sm text-gray-500">No models match.</p>
			{:else}
				<ul class="grid gap-3 md:grid-cols-2">
					{#each filteredCatalog as entry (entry.id)}
						{@const job = jobByCatalog.get(entry.id)}
						<li class="flex flex-col rounded-md border border-gray-200 p-4 dark:border-gray-700">
							<div class="flex items-start justify-between gap-2">
								<div>
									<p class="font-medium">
										{entry.name}
										{#if entry.gated}<span title="Gated: accept licence + token">🔒</span>{/if}
									</p>
									<p class="text-xs text-gray-500 dark:text-gray-400">
										{TYPE_LABELS[entry.type]}{entry.family
											? ` · ${entry.family}`
											: ''}{entry.base_model_family
											? ` · for ${entry.base_model_family}`
											: ''}{entry.license ? ` · ${entry.license}` : ''}
									</p>
								</div>
								{#if entry.homepage}
									<!-- external link to the model page, not an app route -->
									<!-- eslint-disable svelte/no-navigation-without-resolve -->
									<a
										href={entry.homepage}
										target="_blank"
										rel="noreferrer"
										class="shrink-0 text-xs text-indigo-600 hover:underline dark:text-indigo-400"
										>Hub ↗</a
									>
									<!-- eslint-enable svelte/no-navigation-without-resolve -->
								{/if}
							</div>
							{#if entry.description}
								<p class="mt-2 text-sm text-gray-700 dark:text-gray-300">{entry.description}</p>
							{/if}
							<p class="mt-2 text-xs text-gray-500 dark:text-gray-400">
								{entry.size_gb ? `${entry.size_gb.toFixed(1)} GB download` : 'size unknown'}
								{#if entry.min_vram_gb}· {entry.min_vram_gb} GB VRAM without offload{/if}
								{#if entry.defaults?.steps}· {entry.defaults.steps} steps{/if}
							</p>
							{#if entry.notes}
								<details class="mt-1 text-xs text-gray-500 dark:text-gray-400">
									<summary class="cursor-pointer">Notes</summary>
									<p class="mt-1 whitespace-pre-line">{entry.notes}</p>
								</details>
							{/if}
							<div class="mt-auto pt-3">
								{#if entry.installed}
									<span class="text-sm font-medium text-emerald-700 dark:text-emerald-400"
										>✓ Installed</span
									>
								{:else if job}
									<span class="text-sm text-indigo-700 dark:text-indigo-300">
										Downloading… {percent(job) ?? 0}%
									</span>
								{:else}
									<button
										class="rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
										onclick={() =>
											startDownload({ catalog_id: entry.id }, entry.name, entry.size_gb)}
										title={entry.gated && !hasToken ? 'Needs a Hugging Face token' : undefined}
									>
										Download{entry.gated && !hasToken ? ' (token needed)' : ''}
									</button>
								{/if}
							</div>
						</li>
					{/each}
				</ul>
			{/if}
		</section>

		<div class="space-y-6">
			<!-- Add your own -->
			<section
				class="rounded-lg border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800"
			>
				<h3 class="mb-1 text-lg font-semibold">Add from Hugging Face or URL</h3>
				<p class="mb-3 text-xs text-gray-500 dark:text-gray-400">
					A repo like <code>stabilityai/sdxl-turbo</code>, a huggingface.co file link, or a direct
					https link (e.g. Civitai). The type is detected when possible.
				</p>
				<form class="space-y-3" onsubmit={submitCustom}>
					<input
						required
						bind:value={customRef}
						placeholder="owner/repo or https://…"
						aria-label="Repository or URL"
						class="w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm dark:border-gray-600 dark:bg-gray-700"
					/>
					<div class="grid grid-cols-2 gap-2">
						<select
							bind:value={customType}
							aria-label="Model type"
							class="rounded-md border border-gray-300 px-2 py-1.5 text-sm dark:border-gray-600 dark:bg-gray-700"
						>
							<option value="">Type: auto</option>
							{#each Object.entries(TYPE_LABELS) as [value, label] (value)}
								<option {value}>{label}</option>
							{/each}
						</select>
						<input
							bind:value={customFile}
							placeholder="file in repo (optional)"
							aria-label="File in repository"
							class="rounded-md border border-gray-300 px-2 py-1.5 text-sm dark:border-gray-600 dark:bg-gray-700"
						/>
					</div>
					<button
						type="submit"
						disabled={submitting || !customRef.trim()}
						class="w-full rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
					>
						{submitting ? 'Starting…' : 'Download'}
					</button>
				</form>
			</section>

			<!-- Installed -->
			<section
				class="rounded-lg border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800"
			>
				<h3 class="mb-3 text-lg font-semibold">Installed</h3>
				{#if localGroups.length === 0}
					<p class="text-sm text-gray-500">
						Nothing yet. Download a model from the catalog, or copy files into the model folders.
					</p>
				{/if}
				{#each localGroups as group (group.type)}
					<h4 class="mt-3 mb-1 text-xs font-semibold tracking-wide text-gray-500 uppercase">
						{TYPE_LABELS[group.type]}
					</h4>
					<ul class="divide-y divide-gray-100 dark:divide-gray-700">
						{#each group.items as m (m.path)}
							<li class="flex items-center justify-between gap-2 py-1.5 text-sm">
								<div class="min-w-0">
									<p class="truncate" title={m.path}>{m.name}</p>
									<p class="truncate text-xs text-gray-500 dark:text-gray-400">
										{formatBytes(m.size_bytes)}{m.family ? ` · ${m.family}` : ''}{m.source?.repo_id
											? ` · ${m.source.repo_id}`
											: ''}
									</p>
								</div>
								<button
									class="shrink-0 rounded px-2 py-0.5 text-xs text-red-700 hover:bg-red-50 dark:text-red-300 dark:hover:bg-red-950"
									onclick={() => removeLocal(m)}
									aria-label={`Delete ${m.name}`}>Delete</button
								>
							</li>
						{/each}
					</ul>
				{/each}
			</section>
		</div>
	</div>
</div>
