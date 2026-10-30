import { describe, expect, it } from 'vitest';
import { etaSeconds, formatBytes, formatDuration, plainSchedulerName } from './format.js';

describe('formatBytes', () => {
	it('formats binary units', () => {
		expect(formatBytes(0)).toBe('0 B');
		expect(formatBytes(1536)).toBe('1.5 KB');
		expect(formatBytes(7.4 * 1024 ** 3)).toBe('7.4 GB');
	});
	it('handles unknown values', () => {
		expect(formatBytes(null)).toBe('?');
		expect(formatBytes(-1)).toBe('?');
	});
});

describe('durations', () => {
	it('formats seconds', () => {
		expect(formatDuration(12)).toBe('12s');
		expect(formatDuration(200)).toBe('3m 20s');
		expect(formatDuration(3900)).toBe('1h 05m');
	});
	it('computes eta', () => {
		expect(etaSeconds(1000, 500, 100)).toBe(5);
		expect(etaSeconds(null, 0, 100)).toBeNull();
	});
});

describe('plainSchedulerName', () => {
	it('accepts only class names', () => {
		expect(plainSchedulerName('LCMScheduler')).toBe('LCMScheduler');
		expect(plainSchedulerName('EulerDiscreteScheduler(timestep_spacing=trailing)')).toBeNull();
		expect(plainSchedulerName(null)).toBeNull();
	});
});
