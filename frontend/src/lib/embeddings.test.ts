import { describe, expect, it } from 'vitest';
import {
	addToken,
	hasToken,
	normalizePrompt,
	removeToken,
	usedEmbeddings,
	type EmbeddingDetail
} from './embeddings.js';

const detail = (name: string, token: string): EmbeddingDetail => ({
	name,
	token,
	family: 'sd1',
	family_label: 'SD 1.x',
	vectors: 1,
	fits: true,
	error: null
});

describe('embedding tokens', () => {
	it('finds whole tokens only, like the server', () => {
		expect(hasToken('photo, (bad-hands-5:1.2)', 'bad-hands-5')).toBe(true);
		expect(hasToken('bad-hands-50', 'bad-hands-5')).toBe(false);
		expect(hasToken('xEasyNegative', 'EasyNegative')).toBe(false);
		expect(hasToken('embedding:EasyNegative.pt', 'EasyNegative')).toBe(true);
		expect(normalizePrompt('embedding:a.safetensors, embedding:b')).toBe('a, b');
	});

	it('adds and removes tokens as comma-separated parts', () => {
		expect(addToken('', 'EasyNegative')).toBe('EasyNegative');
		expect(addToken('blurry', 'EasyNegative')).toBe('blurry, EasyNegative');
		expect(addToken('blurry, ', 'EasyNegative')).toBe('blurry, EasyNegative');
		expect(removeToken('blurry, EasyNegative, low quality', 'EasyNegative')).toBe(
			'blurry, low quality'
		);
		expect(removeToken('(EasyNegative:1.2), blurry', 'EasyNegative')).toBe('blurry');
		expect(removeToken('embedding:EasyNegative.pt', 'EasyNegative')).toBe('');
	});

	it('lists the embeddings used in either prompt', () => {
		const installed = [
			detail('neg.pt', 'neg'),
			detail('neg.safetensors', 'neg'),
			detail('st.pt', 'st')
		];
		const used = usedEmbeddings('a cat, st', 'neg', installed);
		expect(used.map((u) => [u.embedding.name, u.where])).toEqual([
			['neg.pt', 'negative_prompt'],
			['st.pt', 'prompt']
		]);
	});
});
