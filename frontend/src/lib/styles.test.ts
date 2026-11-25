import { describe, expect, it } from 'vitest';
import { groupStyles, needsPreview, tokenSummary, type PromptStyle } from './styles.js';

const style = (name: string, group: string, prompt = ''): PromptStyle => ({
	name,
	group,
	prompt,
	negative_prompt: '',
	file: 'styles.csv'
});

describe('styles', () => {
	const all = [
		style('CX: Dynamic', 'CX', 'dynamic pose'),
		style('PX: Analog', 'PX', 'film grain'),
		style('CX: Close', 'CX', 'close-up')
	];

	it('groups and searches styles', () => {
		expect(groupStyles(all).map(([g, list]) => [g, list.length])).toEqual([
			['CX', 2],
			['PX', 1]
		]);
		expect(groupStyles(all, 'grain')).toEqual([['PX', [all[1]]]]);
	});

	it('summarises token counts', () => {
		expect(tokenSummary({ tokens: 92, chunks: 2, limit: 75 }, false)).toBe(
			'92 tokens · 2 chunks of 75'
		);
		expect(tokenSummary({ tokens: 92, chunks: 2, limit: 75 }, true)).toBe('92 tokens: cut at 75');
		expect(tokenSummary({ tokens: 1, chunks: 1, limit: 75 }, false)).toBe('1 token');
		expect(tokenSummary(null, false)).toBe('');
	});

	it('asks for a preview only when it can differ from the prompt', () => {
		expect(needsPreview('a cat', '', [])).toBe(false);
		expect(needsPreview('a cat', '', ['CX: Dynamic'])).toBe(true);
		expect(needsPreview('a {red|blue} cat', '', null)).toBe(true);
		expect(needsPreview('a cat <lora:ink:0.7>', '', null)).toBe(true);
		expect(needsPreview('{prompt} only', '', null)).toBe(false);
	});
});
