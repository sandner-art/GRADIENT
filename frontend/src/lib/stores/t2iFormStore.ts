// frontend/src/lib/stores/t2iFormStore.ts
import { writable } from 'svelte/store';
import type { Text2ImageRequest } from '$lib/types';

export const t2iDefaultState: Text2ImageRequest = {
    model_id: 'runwayml/stable-diffusion-v1-5',
    prompt: 'photo of a tabby cat programmer writing code, high resolution',
    negative_prompt: 'ugly, blurry, deformed, text, words, low quality',
    height: 512,
    width: 512,
    num_inference_steps: 25,
    guidance_scale: 7.0,
    seed: null,
    scheduler_id: null,
    num_outputs: 1,
    custom_sigmas: null
};

export const t2iFormStore = writable<Text2ImageRequest>({ ...t2iDefaultState });

export function resetT2iFormStore() {
    t2iFormStore.set({ ...t2iDefaultState });
}