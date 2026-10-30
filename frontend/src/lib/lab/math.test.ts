import { describe, expect, it } from 'vitest';
import {
	attentionMethodFor,
	cleanLane,
	composeTaylor,
	countMatches,
	globToRegExp,
	laneProblem,
	laneSteps,
	meanAbsDiff,
	parseWordList
} from './math.js';

const px = (...values: number[]) => new Uint8ClampedArray(values);

describe('composeTaylor', () => {
	it('returns the base image at delta 0', () => {
		const base = px(10, 20, 30, 255);
		expect([...composeTaylor(base, px(255, 0, 128, 255), 0.1, 0)]).toEqual([10, 20, 30, 255]);
	});
	it('adds delta * tangent in 0-255 units', () => {
		// tangent value 255 -> t = +scale; scale 0.1 per unit CFG -> +25.5 per unit
		const out = composeTaylor(px(100, 100, 100, 255), px(255, 1, 128, 255), 0.1, 2);
		expect([...out]).toEqual([151, 49, 100, 255]);
	});
	it('clamps to the valid range', () => {
		const out = composeTaylor(px(250, 5, 0, 255), px(255, 1, 128, 255), 1, 1);
		expect([...out]).toEqual([255, 0, 0, 255]);
	});
});

describe('meanAbsDiff', () => {
	it('ignores alpha', () => {
		expect(meanAbsDiff(px(0, 0, 0, 0), px(3, 6, 9, 255))).toBe(6);
		expect(meanAbsDiff(px(1, 2, 3, 4), px(1, 2))).toBeNaN();
	});
});

describe('parseWordList', () => {
	it('splits on commas and trims', () => {
		expect(parseWordList(' cat, red  hat,,moon ')).toEqual(['cat', 'red hat', 'moon']);
		expect(parseWordList('  , ')).toBeNull();
	});
});

describe('lanes', () => {
	it('maps fractions to steps like the backend', () => {
		expect(laneSteps(0, 0.3, 20)).toEqual([0, 6]);
		expect(laneSteps(0.25, 0.5, 30)).toEqual([7, 15]);
		expect(laneSteps(0, 1, 7)).toEqual([0, 7]);
	});
	it('validates lanes', () => {
		expect(laneProblem({ type: 'guidance', start: 0.5, end: 0.4, value: 3 }, 0)).toMatch(/end/);
		expect(laneProblem({ type: 'guidance', start: 0, end: 1 }, 0)).toMatch(/value/);
		expect(laneProblem({ type: 'prompt', start: 0, end: 0.3, prompt: ' ' }, 0)).toMatch(/prompt/);
		expect(
			laneProblem({ type: 'attention_scale', start: 0, end: 1, value: 0, targets: '' }, 0)
		).toMatch(/target/);
		expect(
			laneProblem({ type: 'lora_weight', start: 0, end: 1, value: 1, lora_index: 0 }, 0)
		).toMatch(/LoRA/);
		expect(laneProblem({ type: 'prompt', start: 0, end: 0.3, prompt: 'a dog' }, 0)).toBeNull();
	});
	it('drops fields the lane type does not use', () => {
		expect(
			cleanLane({
				type: 'guidance',
				start: 0,
				end: 0.5,
				value: 9,
				prompt: 'x',
				targets: '*',
				lora_index: 1
			})
		).toEqual({ type: 'guidance', start: 0, end: 0.5, value: 9 });
		expect(cleanLane({ type: 'prompt', start: 0, end: 1, prompt: 'p', mode: 'blend' })).toEqual({
			type: 'prompt',
			start: 0,
			end: 1,
			prompt: 'p',
			mode: 'blend',
			value: 0.5
		});
	});
});

describe('layer globs', () => {
	const layers = {
		model_id: 'm',
		pipeline: 'StableDiffusionPipeline',
		suggestions: [],
		concept_blocks: 0,
		layers: [
			{ name: 'down_blocks.0.attentions.0.transformer_blocks.0.attn1', kind: 'self' },
			{ name: 'down_blocks.0.attentions.0.transformer_blocks.0.attn2', kind: 'cross' },
			{ name: 'up_blocks.1.attentions.2.transformer_blocks.0.attn2', kind: 'cross' },
			{ name: 'transformer_blocks.12.attn', kind: 'joint' }
		]
	};
	it('matches like fnmatch', () => {
		expect(countMatches(layers, '*attn2')).toBe(2);
		expect(countMatches(layers, 'up_blocks.*attn2, *attn1')).toBe(2);
		expect(countMatches(layers, 'transformer_blocks.1?.attn')).toBe(1);
		expect(countMatches(layers, 'down_blocks.[!1].*')).toBe(2);
		expect(countMatches(layers, 'nothing')).toBe(0);
		expect(countMatches(null, '*')).toBeNull();
		expect(globToRegExp('a.b(c)').test('a.b(c)')).toBe(true);
		expect(globToRegExp('a.b').test('aXb')).toBe(false);
	});
});

describe('attentionMethodFor', () => {
	it('picks ConceptAttention for DiTs and cross-attention for UNets', () => {
		expect(attentionMethodFor('flux1')).toBe('concept');
		expect(attentionMethodFor('flux2')).toBe('concept');
		expect(attentionMethodFor('sd3')).toBe('concept');
		expect(attentionMethodFor('sdxl')).toBe('cross');
		expect(attentionMethodFor('qwen-image')).toBeNull();
		expect(attentionMethodFor(null)).toBeNull();
	});
});
