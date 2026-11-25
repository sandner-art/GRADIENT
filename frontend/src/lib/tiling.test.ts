import { describe, expect, it } from 'vitest';
import { seamScores, seamVerdict } from './tiling.js';

function image(w: number, h: number, f: (x: number, y: number) => number): Uint8ClampedArray {
	const data = new Uint8ClampedArray(w * h * 4);
	for (let y = 0; y < h; y++)
		for (let x = 0; x < w; x++) {
			const v = f(x, y);
			data.set([v, v, v, 255], (y * w + x) * 4);
		}
	return data;
}

describe('seam scores', () => {
	it('finds no seam in a periodic pattern and a seam in a ramp', () => {
		const w = 32,
			h = 16;
		const wave = image(w, h, (x, y) => 128 + 100 * Math.sin((2 * Math.PI * x) / w) + y * 0);
		const s = seamScores(wave, w, h);
		expect(s.x).toBeLessThan(1.3);
		const ramp = image(w, h, (x) => x * 8); // jumps from 248 back to 0 at the wrap
		expect(seamScores(ramp, w, h).x).toBeGreaterThan(10);
		expect(seamVerdict(1.1)).toBe('no visible seam');
		expect(seamVerdict(1.7)).toBe('faint seam');
		expect(seamVerdict(5)).toBe('visible seam');
	});

	it('scores the two axes separately', () => {
		const w = 16,
			h = 16;
		const rows = image(w, h, (_x, y) => y * 16); // seam only top-bottom
		const s = seamScores(rows, w, h);
		expect(s.x).toBe(1); // flat left-right: nothing to compare, no seam
		expect(s.y).toBeGreaterThan(10);
	});
});
