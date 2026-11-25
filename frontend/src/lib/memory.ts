// frontend/src/lib/memory.ts
// GET /system/memory and POST /system/unload.

export interface LoadedItem {
	name: string;
	detail?: string;
	vram_mb?: number | null;
}

export interface Vram {
	total_gb: number;
	free_gb: number;
	allocated_gb: number;
	reserved_gb: number;
}

export interface MemoryReport {
	device: string;
	vram: Vram | null;
	loaded: { diffusion: LoadedItem[]; vision: LoadedItem[]; llm: LoadedItem[] };
	caches: Record<string, number>;
}

export interface UnloadResult {
	unloaded: number;
	freed: {
		diffusion?: string[];
		vision?: string[];
		llm?: string[];
		caches?: Record<string, number>;
	};
	vram_before: Vram | null;
	vram_after: Vram | null;
}

/** "Freed 5.3 GB: 2 image models, 1 vision model" (or what was unloaded, without CUDA). */
export function freedMessage(r: UnloadResult): string {
	const parts: string[] = [];
	const count = (n: number, one: string) => `${n} ${one}${n === 1 ? '' : 's'}`;
	if (r.freed.diffusion?.length) parts.push(count(r.freed.diffusion.length, 'image model'));
	if (r.freed.vision?.length) parts.push(count(r.freed.vision.length, 'vision model'));
	if (r.freed.llm?.length) parts.push(count(r.freed.llm.length, 'LLM'));
	if (r.freed.caches) parts.push('caches');
	const what = parts.length ? parts.join(', ') : 'nothing was loaded';
	if (r.vram_before && r.vram_after) {
		const gb = r.vram_after.free_gb - r.vram_before.free_gb;
		if (gb >= 0.05) return `Freed ${gb.toFixed(1)} GB of VRAM: ${what}`;
	}
	return parts.length ? `Unloaded ${what}` : 'Nothing was loaded';
}
