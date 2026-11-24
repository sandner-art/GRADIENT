import { describe, expect, it } from 'vitest';
import { loraState, sweepProgress } from './sweeps.js';

describe('loraState', () => {
	it('accepts matching and name-only LoRAs, refuses missing and different files', () => {
		expect(loraState('ok')).toEqual({ mark: '✓', ok: true });
		expect(loraState('found_by_hash').ok).toBe(true);
		expect(loraState('unverified')).toEqual({ mark: '?', ok: true });
		expect(loraState('missing').ok).toBe(false);
		expect(loraState('hash_mismatch')).toEqual({ mark: '✗', ok: false });
	});
});

describe('sweepProgress', () => {
	it('counts finished jobs and names failures', () => {
		expect(sweepProgress({ images: 10, counts: { completed: 4, queued: 6 } })).toBe('4/10 done');
		expect(sweepProgress({ images: 5, counts: { completed: 2, failed: 1, cancelled: 2 } })).toBe(
			'5/5 done · 1 failed · 2 cancelled'
		);
	});
});
