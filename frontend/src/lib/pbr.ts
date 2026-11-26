// frontend/src/lib/pbr.ts
// PBR map sets (POST /pbr/maps) and the lit preview: an albedo lit by a normal map, with a
// light the pointer moves. Shading is plain Lambert plus ambient, times AO.

import { get, post } from '#lib/apiClient.js';

export type PbrMethod = 'hybrid' | 'network' | 'classical';
export type Seamless = 'auto' | 'none' | 'x' | 'y' | 'xy';

export interface PbrModels {
	folder: string;
	models: { name: string; size_mb: number }[];
	default: string | null;
}

export interface MapFile {
	kind: string;
	label: string;
	file: string;
	color: boolean;
}

export interface MapSet {
	set_id: string;
	name: string;
	width: number;
	height: number;
	method: PbrMethod;
	model: string | null;
	seamless: { x: boolean; y: boolean };
	maps: MapFile[];
	sidecar: string;
	light: { azimuth_deg: number; elevation_deg: number; direction: number[] } | null;
	height_range_px: number;
	notes: string[];
	seconds: number;
	source_output_id: number | null;
}

export interface MapsRequest {
	output_id?: number;
	image_base64?: string;
	name?: string;
	model?: string;
	method?: PbrMethod;
	seamless?: Seamless;
	detail?: number;
	strength?: number;
}

export const pbrModels = () => get<PbrModels>('/pbr/models');
export const makeMaps = (req: MapsRequest) => post<MapSet>('/pbr/maps', req);
export const mapFileUrl = (setId: string, file: string) =>
	`/api/v1/pbr/maps/${setId}/files/${encodeURIComponent(file)}`;
export const mapZipUrl = (setId: string) => `/api/v1/pbr/maps/${setId}/zip`;

export function mapOf(set: MapSet, kind: string): MapFile | undefined {
	return set.maps.find((m) => m.kind === kind);
}

/** "Seamless left–right", "Seamless both ways", "Not seamless" */
export function seamlessLabel(s: { x: boolean; y: boolean }): string {
	if (s.x && s.y) return 'Seamless both ways';
	if (s.x) return 'Seamless left–right';
	if (s.y) return 'Seamless top–bottom';
	return 'Not seamless';
}

/** Light direction (unit, x right, y up, z towards the viewer) from a pointer position over an
 *  image of w × h: the light sits over the pointer, `elevation` (0..1) above the surface. */
export function lightFromPointer(
	x: number,
	y: number,
	w: number,
	h: number,
	elevation = 0.5
): [number, number, number] {
	const r = Math.max(w, h) / 2;
	const lx = (x - w / 2) / r;
	const ly = -(y - h / 2) / r; // screen y grows downwards
	const lz = Math.max(0.05, elevation) * 2;
	const n = Math.hypot(lx, ly, lz) || 1;
	return [lx / n, ly / n, lz / n];
}

/** Light a colour image with a normal map (OpenGL: green up), RGBA arrays of the same size.
 *  `ao` (optional) darkens the ambient and direct light alike. Writes into `out`. */
export function shade(
	color: Uint8ClampedArray,
	normal: Uint8ClampedArray,
	ao: Uint8ClampedArray | null,
	out: Uint8ClampedArray,
	light: [number, number, number],
	ambient = 0.25,
	gain = 1.2
): Uint8ClampedArray {
	const [lx, ly, lz] = light;
	for (let i = 0; i < color.length; i += 4) {
		let nx = normal[i] / 127.5 - 1;
		let ny = normal[i + 1] / 127.5 - 1;
		let nz = normal[i + 2] / 127.5 - 1;
		const len = Math.hypot(nx, ny, nz) || 1;
		nx /= len;
		ny /= len;
		nz /= len;
		const lambert = Math.max(0, nx * lx + ny * ly + nz * lz);
		const occ = ao ? ao[i] / 255 : 1;
		const k = (ambient + (1 - ambient) * lambert) * occ * gain;
		out[i] = color[i] * k;
		out[i + 1] = color[i + 1] * k;
		out[i + 2] = color[i + 2] * k;
		out[i + 3] = 255;
	}
	return out;
}
