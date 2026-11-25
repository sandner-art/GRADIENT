import { describe, expect, it } from 'vitest';
import { freedMessage, type UnloadResult, type Vram } from './memory.js';

const vram = (free: number): Vram => ({
	total_gb: 12,
	free_gb: free,
	allocated_gb: 0,
	reserved_gb: 0
});

describe('freedMessage', () => {
	it('names what was unloaded and the VRAM freed', () => {
		const r: UnloadResult = {
			unloaded: 2,
			freed: { diffusion: ['a', 'b'], vision: ['clip'], llm: [] },
			vram_before: vram(2),
			vram_after: vram(7.3)
		};
		expect(freedMessage(r)).toBe('Freed 5.3 GB of VRAM: 2 image models, 1 vision model');
	});

	it('works without CUDA figures and when nothing was loaded', () => {
		const base = { unloaded: 0, vram_before: null, vram_after: null };
		expect(freedMessage({ ...base, freed: { vision: ['dino'] } })).toBe('Unloaded 1 vision model');
		expect(freedMessage({ ...base, freed: { diffusion: [], vision: [] } })).toBe(
			'Nothing was loaded'
		);
	});
});
