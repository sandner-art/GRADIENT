<!-- frontend/src/lib/components/render/ExplorePane.svelte -->
<!-- The Explore mode of the preview panel: explore the shown image right beside the form
     (same session as the Lab's Explorer, which has the full map and options). -->
<script lang="ts">
	import { get as read } from 'svelte/store';
	import Explorer from '#lib/components/lab/Explorer.svelte';
	import { getOutput, thumbUrl } from '#lib/gallery/api.js';
	import { explore, loadImage, openFromParams } from '#lib/lab/explore.svelte.js';
	import { lab } from '#lib/lab/labState.svelte.js';
	import * as ui from '#lib/lab/ui.js';
	import { i2iFormStore } from '#lib/stores/i2iFormStore.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	let { form, id }: { form: 'text' | 'image'; id: number | null } = $props();

	const source = $derived(id === null ? null : `image #${id}`);
	const active = $derived(source !== null && explore.session?.source === source);

	async function open() {
		if (id === null || source === null) return;
		explore.error = null;
		try {
			const detail = await getOutput(id);
			const init = form === 'image' ? (read(i2iFormStore).image_base64 ?? null) : null;
			const problem = openFromParams(detail.parameters, source, init);
			if (problem) {
				explore.error = problem;
				return;
			}
			await loadImage();
		} catch (e) {
			explore.error = apiErrorMessage(e);
		}
	}
</script>

{#if active}
	<Explorer compact />
{:else}
	<div class="{ui.card} !p-4 text-center">
		{#if id === null}
			<p class={ui.muted}>Render an image, then explore it here.</p>
		{:else}
			<img src={thumbUrl(id, 256)} alt="" class="mx-auto mb-3 h-40 rounded object-contain" />
			<p class="{ui.muted} mx-auto max-w-md">
				Mark a region and branch it from any step: new variations, other words, while the rest of
				the image stays.
			</p>
			<button
				type="button"
				class="{ui.primaryButton} mt-3"
				disabled={lab.busy !== null}
				onclick={open}>Explore this image</button
			>
			{#if explore.error}<div class="{ui.errorBox} mt-3 text-left" role="alert">
					{explore.error}
				</div>{/if}
			{#if explore.session}
				<p class="{ui.help} mt-2">The Lab's Explorer has {explore.session.source} open.</p>
			{/if}
		{/if}
	</div>
{/if}
