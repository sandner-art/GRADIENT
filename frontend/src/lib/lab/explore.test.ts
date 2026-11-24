// frontend/src/lib/lab/explore.test.ts
import { describe, expect, it } from 'vitest';
import {
	explore,
	openFromParams,
	pathFromParams,
	replaceWord,
	requestFromParams
} from './explore.svelte.js';

describe('replaceWord', () => {
	it('replaces whole words only, keeping the rest', () => {
		expect(replaceWord('a red lighthouse, red sky', 'red', 'blue')).toBe(
			'a blue lighthouse, red sky'
		);
		expect(replaceWord('a reddish lighthouse', 'red', 'blue')).toBe('a reddish lighthouse');
		expect(replaceWord('Red brick wall', 'red', 'white')).toBe('white brick wall');
	});
});

describe('explorer sessions from saved images', () => {
	const params = {
		model_id: 'sd15',
		prompt: 'a lighthouse',
		seed: 7,
		guidance_scale: '7 -> 3',
		quantization: 'none',
		width: 512,
		lab_tool: 'explore_branches',
		explore_path: [{ step: 4, kind: 'seed', seed: 1007, mask_base64: null }]
	};

	it('builds a render request and keeps the branch path', () => {
		const req = requestFromParams(params);
		expect(req).toMatchObject({ model_id: 'sd15', prompt: 'a lighthouse', seed: 7, width: 512 });
		expect(req.guidance_scale).toBeUndefined(); // scheduled guidance is not a number
		expect(req.quantization).toBeNull();
		expect(pathFromParams(params)).toHaveLength(1);
		expect(pathFromParams({})).toEqual([]);
	});

	it('opens a session, continuing the path', () => {
		expect(openFromParams(params, 'image #3')).toBeNull();
		expect(explore.session?.source).toBe('image #3');
		expect(explore.session?.path[0].seed).toBe(1007);
		expect(explore.start).toBeNull();
	});

	it('needs the input image for image-to-image renders', () => {
		const i2i = { ...params, strength: 0.6, explore_path: undefined };
		expect(openFromParams(i2i, 'image #4')).toMatch(/Image-to-Image/);
		expect(openFromParams(i2i, 'image #4', 'data:image/png;base64,AAAA')).toBeNull();
		expect(explore.session?.strength).toBe(0.6);
		expect(explore.session?.init_image_base64).toContain('base64');
		expect(openFromParams({ prompt: 'x' }, 'image #5')).toMatch(/no saved settings/);
	});
});
