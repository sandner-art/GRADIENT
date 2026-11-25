// frontend/src/lib/gallery/gallery.test.ts
import { describe, expect, it } from 'vitest';

import { exportFilter, filterQuery, runTypeLabel, tagSourceLabel } from './api.js';
import {
	applyEvent,
	jobFraction,
	onJobChange,
	outputIds,
	queue
} from '#lib/stores/queue.svelte.js';
import { settingsFrom } from '#lib/stores/ui.svelte.js';
import type { QueueJob } from '#lib/types.js';

function job(over: Partial<QueueJob> = {}): QueueJob {
	return {
		id: 'a',
		kind: 'text_to_image',
		label: 'test',
		status: 'queued',
		priority: 0,
		created: 0,
		started: null,
		finished: null,
		progress: {},
		run_uuid: null,
		error: null,
		result: null,
		position: null,
		retryable: false,
		model_id: null,
		...over
	};
}

describe('filterQuery', () => {
	it('drops empty filters and keeps paging', () => {
		const q = new URLSearchParams(
			filterQuery({ q: 'cat', model: null, favorite: false, min_rating: 3, tag: '' }, 60, 30)
		);
		expect(Object.fromEntries(q)).toEqual({ q: 'cat', min_rating: '3', offset: '60', limit: '30' });
	});
	it('passes true flags as strings', () => {
		expect(filterQuery({ favorite: true })).toContain('favorite=true');
	});
});

describe('exportFilter', () => {
	it('keeps only real filters (no sort, no empty values)', () => {
		expect(
			exportFilter({
				sort: 'rating',
				q: 'cat',
				model: null,
				favorite: false,
				tag: '',
				project: 'p1'
			})
		).toEqual({ q: 'cat', project: 'p1' });
	});
});

describe('tagSourceLabel', () => {
	it('explains where a tag came from', () => {
		expect(tagSourceLabel('manual')).toBe('yours');
		expect(tagSourceLabel('xmp')).toContain('Kurator');
		expect(tagSourceLabel('api')).toContain('ArtAgents');
		expect(tagSourceLabel('artagents:llava')).toBe('from artagents:llava');
	});
});

describe('runTypeLabel', () => {
	it('names lab runs and falls back to the raw type', () => {
		expect(runTypeLabel('lab_taylor')).toBe('Lab · Taylor');
		expect(runTypeLabel('group')).toBe('Group');
		expect(runTypeLabel('custom')).toBe('custom');
		expect(runTypeLabel(null)).toBe('Unknown');
	});
});

describe('jobFraction', () => {
	it('uses steps for a single render', () => {
		expect(jobFraction(job({ progress: { step: 5, total: 20 } }))).toBe(0.25);
		expect(jobFraction(job())).toBeNull();
	});
	it('combines item and step for grids', () => {
		// item 3 of 4, half way through its steps -> (2 + 0.5) / 4
		expect(jobFraction(job({ progress: { item: 3, items: 4, step: 10, total: 20 } }))).toBe(0.625);
	});
});

describe('outputIds', () => {
	it('skips outputs that were not logged', () => {
		const j = job({
			result: {
				outputs: [
					{ id: 4, seed: 1 },
					{ id: null, seed: 2 },
					{ id: 7, seed: 3 }
				]
			}
		});
		expect(outputIds(j)).toEqual([4, 7]);
		expect(outputIds(undefined)).toEqual([]);
	});
});

describe('applyEvent', () => {
	it('keeps the waiting order and notifies listeners of status changes', () => {
		applyEvent({
			type: 'snapshot',
			jobs: [
				job({ id: 'b', position: 2 }),
				job({ id: 'a', position: 1 }),
				job({ id: 'c', status: 'running' })
			]
		});
		expect(queue.order).toEqual(['a', 'b']);

		const seen: string[] = [];
		const off = onJobChange((j, prev) => seen.push(`${j.id}:${prev?.status}->${j.status}`));
		applyEvent({ type: 'job', job: job({ id: 'a', status: 'running' }) });
		expect(queue.order).toEqual(['b']);
		expect(seen).toEqual(['a:queued->running']);

		applyEvent({ type: 'order', order: ['zzz', 'b'] });
		expect(queue.order).toEqual(['b']);
		applyEvent({ type: 'removed', ids: ['c'] });
		expect(queue.jobs.c).toBeUndefined();
		off();
	});
});

describe('settingsFrom', () => {
	const params = {
		model_id: 'sd15',
		prompt: 'a cat',
		seed: 42,
		guidance_scale: '7->3',
		loras: [{ name: 'ink', weight: 0.7 }],
		quantization: 'bnb-4bit',
		run_type: 'text_to_image'
	};
	it('copies form settings and drops scheduled CFG strings', () => {
		const s = settingsFrom(params, false);
		expect(s).toEqual({
			model_id: 'sd15',
			prompt: 'a cat',
			loras: [{ name: 'ink', weight: 0.7 }],
			quantization: 'bnb-4bit',
			prompt_syntax: 'plain', // saved before 0.10: prompts were read as plain text
			seed: null
		});
	});
	it('keeps the seed when asked', () => {
		expect(settingsFrom(params, true).seed).toBe(42);
	});
	it('keeps the prompt syntax the image was rendered with', () => {
		expect(settingsFrom({ ...params, prompt_syntax: 'comfy' }, false).prompt_syntax).toBe('comfy');
	});
});
