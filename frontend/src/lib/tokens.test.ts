import { describe, expect, it } from 'vitest';
import { tokenLine, weightStyle, type TokenView } from './tokens.js';

const view = (n: number, chunks: number, dropped = 0): TokenView => ({
	tokens: n,
	limit: 75,
	chunks: Array.from({ length: chunks }, () => []),
	dropped: Array.from({ length: dropped }, () => ({ t: 'x', w: 1, end: true, emb: false })),
	edited: false
});

describe('token view', () => {
	it('summarises counts, chunks and what plain text cuts off', () => {
		expect(tokenLine(view(23, 1), false)).toBe('23');
		expect(tokenLine(view(92, 2), false)).toBe('92 · 2 chunks');
		expect(tokenLine(view(92, 1, 17), true)).toBe('92: 17 cut off');
	});

	it('shows weights as colour', () => {
		expect(weightStyle(1)).toBe('');
		expect(weightStyle(1.3)).toContain('0.27');
		expect(weightStyle(2)).toContain('0.55'); // capped
		expect(weightStyle(0.5)).toBe('opacity: 0.50');
	});
});
