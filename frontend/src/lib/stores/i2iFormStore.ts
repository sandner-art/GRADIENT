// frontend/src/lib/stores/i2iFormStore.ts
import { writable } from 'svelte/store';
import type { ImageToImageRequest } from '$lib/types';

export const i2iDefaultState: ImageToImageRequest = {
    model_id: 'photomatix_v3.fp16.safetensors',
    prompt: '',
    negative_prompt: '',
    height: 512,
    width: 512,
    num_inference_steps: 35,
    guidance_scale: 7.0,
    seed: null,
    scheduler_id: null,
    num_outputs: 1,
    image_base64: '',
    strength: 0.75
};

// Note: image_base64 is transient, might not need reset? Or reset to ''?
export const i2iFormStore = writable<ImageToImageRequest>({ ...i2iDefaultState });

 export function resetI2iFormStore() {
    // Reset everything EXCEPT maybe the base64 image? Or clear it too?
    // Let's clear it for now.
    i2iFormStore.set({ ...i2iDefaultState, image_base64: '' });
}