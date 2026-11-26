import { describe, expect, it } from 'vitest';
import { lightFromPointer, mapFileUrl, seamlessLabel, shade } from './pbr.js';

const px = (...v: number[]) => new Uint8ClampedArray(v);

describe('lit preview', () => {
	it('puts the light over the pointer, y up', () => {
		const [x, y, z] = lightFromPointer(50, 50, 100, 100);
		expect(x).toBeCloseTo(0);
		expect(y).toBeCloseTo(0);
		expect(z).toBeCloseTo(1);
		const upLeft = lightFromPointer(0, 0, 100, 100);
		expect(upLeft[0]).toBeLessThan(0);
		expect(upLeft[1]).toBeGreaterThan(0);
		expect(Math.hypot(...upLeft)).toBeCloseTo(1);
	});

	it('lights a flat normal fully from above and a tilted one by its angle', () => {
		const color = px(200, 100, 50, 255, 200, 100, 50, 255);
		// flat (0,0,1) and tilted to the right (+x), as OpenGL normal map colours
		const normal = px(128, 128, 255, 255, 218, 128, 218, 255);
		const out = new Uint8ClampedArray(8);
		shade(color, normal, null, out, [0, 0, 1], 0, 1);
		expect(out[0]).toBeGreaterThan(195);
		expect(out[4]).toBeLessThan(160);
		// light from the right: the tilted pixel is the brighter one
		shade(color, normal, null, out, [1, 0, 0], 0, 1);
		expect(out[4]).toBeGreaterThan(out[0]);
		// ambient occlusion darkens
		shade(color, normal, px(255, 0, 0, 255, 64, 0, 0, 255), out, [0, 0, 1], 0.25, 1);
		expect(out[4]).toBeLessThan(out[0] / 2);
	});
});

describe('labels and urls', () => {
	it('describes the wrap axes and escapes file names', () => {
		expect(seamlessLabel({ x: true, y: true })).toBe('Seamless both ways');
		expect(seamlessLabel({ x: false, y: true })).toBe('Seamless top–bottom');
		expect(seamlessLabel({ x: false, y: false })).toBe('Not seamless');
		expect(mapFileUrl('ab', 'a b.png')).toBe('/api/v1/pbr/maps/ab/files/a%20b.png');
	});
});
