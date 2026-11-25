import { describe, expect, it } from 'vitest';
import { axisMax, labelOf, parseStarts, sortRows, type StyleCheckRow } from './styleCheck.js';

const row = (style: string, s: number, c: number | null): StyleCheckRow => ({
	style,
	position: 'template',
	start: 0,
	strength: s / 10,
	strength_rel: s,
	content: null,
	content_rel: c,
	consistency: null,
	tokens: null,
	chunks: null,
	notes: [],
	cells: []
});

describe('style check helpers', () => {
	it('sorts by strength or content kept', () => {
		const rows = [row('a', 0.4, 0.9), row('b', 1.2, 0.7), row('c', 0.8, null)];
		expect(sortRows(rows, 'strength').map((r) => r.style)).toEqual(['b', 'c', 'a']);
		expect(sortRows(rows, 'content').map((r) => r.style)).toEqual(['a', 'b', 'c']);
		expect(sortRows(rows, 'order')).toBe(rows);
	});

	it('parses style starts', () => {
		expect(parseStarts('0, 0.3, 0.3, x, 1.5')).toEqual([0, 0.3]);
		expect(parseStarts('')).toEqual([0]);
		expect(parseStarts('0.3,')).toEqual([0.3]);
	});

	it('scales axes to cover the reference value 1', () => {
		expect(axisMax([0.3, 0.5])).toBe(1.2);
		expect(axisMax([2.1])).toBe(2.5);
		expect(labelOf({ ...row('PX: Analog', 1, 1), start: 0.3 })).toBe(
			'PX: Analog · template · from 0.3'
		);
	});
});
