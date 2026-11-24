// frontend/src/lib/stores/nav.test.ts
import { describe, expect, it } from 'vitest';

import { closedOverlay, samePlace, tabFromHash, type NavState } from './nav.svelte.js';
import { sendToLab, ui } from './ui.svelte.js';
import { lab } from '#lib/lab/labState.svelte.js';
import { jobMenu } from '#lib/gallery/menus.js';
import type { QueueJob } from '#lib/types.js';

const base: NavState = { tab: 'gallery', view: 'images', viewer: null, queue: false };

describe('navigation history', () => {
	it('treats paging inside the viewer as the same place', () => {
		const a = { ...base, viewer: { ids: [1, 2, 3], index: 0 } };
		expect(samePlace(a, { ...a, viewer: { ids: [1, 2, 3], index: 2 } })).toBe(true);
		expect(samePlace(a, base)).toBe(false);
		expect(samePlace(base, { ...base, tab: 't2i' })).toBe(false);
		expect(samePlace(undefined, base)).toBe(false);
	});
	it('knows when an overlay was closed', () => {
		expect(closedOverlay({ ...base, viewer: { ids: [1], index: 0 } }, base)).toBe(true);
		expect(closedOverlay({ ...base, queue: true }, { ...base, tab: 't2i' })).toBe(true);
		expect(closedOverlay(base, { ...base, queue: true })).toBe(false);
	});
	it('reads the tab from the URL', () => {
		expect(tabFromHash('#lab')).toBe('lab');
		expect(tabFromHash('#nope')).toBeNull();
	});
});

describe('sendToLab', () => {
	it('copies settings, keeps one LoRA and says what changed', () => {
		sendToLab({
			model_id: 'sd15',
			prompt: 'a fox',
			width: 768,
			height: '512',
			num_inference_steps: 30,
			guidance_scale: 'linear(9,3)',
			seed: 42,
			loras: [
				{ name: 'ink.safetensors', weight: 0.6 },
				{ name: 'paper.safetensors', weight: 1 }
			]
		});
		expect(ui.tab).toBe('lab');
		expect(lab.base).toMatchObject({
			model_id: 'sd15',
			prompt: 'a fox',
			width: 768,
			height: 512,
			num_inference_steps: 30,
			seed: 42,
			lora: 'ink.safetensors',
			lora_weight: 0.6
		});
		expect(typeof lab.base.guidance_scale).toBe('number');
		expect(ui.toasts.at(-1)?.text).toContain('one LoRA');
	});
});

describe('jobMenu', () => {
	const job = (over: Partial<QueueJob>): QueueJob => ({
		id: 'j',
		kind: 'text_to_image',
		label: 'a fox',
		status: 'queued',
		priority: 0,
		created: 0,
		started: null,
		finished: null,
		progress: {},
		run_uuid: null,
		error: null,
		result: null,
		position: 1,
		retryable: false,
		model_id: 'sd15',
		payload: { model_id: 'sd15', prompt: 'a fox' },
		...over
	});
	const labels = (j: QueueJob) => jobMenu(j).flatMap((i) => ('label' in i ? [i.label] : []));
	it('offers what fits the job', () => {
		expect(labels(job({}))).toEqual(
			expect.arrayContaining(['Run next', 'Use settings', 'Send to Lab', 'Cancel'])
		);
		expect(labels(job({ status: 'running' }))).toContain('Stop');
		const failed = labels(job({ status: 'failed', retryable: true }));
		expect(failed).toContain('Retry');
		expect(failed).not.toContain('Cancel');
	});
});
