// frontend/src/lib/tiling.ts
// Seam scores of a (meant to be) tileable image: how much the colour jumps across the wrap
// edge compared with an ordinary step between neighbouring pixels. 1 = no seam.

export type Tiling = 'none' | 'x' | 'y' | 'xy';

export const TILINGS: { value: Tiling; label: string }[] = [
	{ value: 'none', label: 'Off' },
	{ value: 'xy', label: 'Seamless both ways' },
	{ value: 'x', label: 'Seamless left–right' },
	{ value: 'y', label: 'Seamless top–bottom' }
];

function luma(data: ArrayLike<number>, i: number): number {
	return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
}

/** RGBA pixels (canvas ImageData) -> {x: left-right seam, y: top-bottom seam}: the jump
 *  across the wrap edge against the steps right next to it (two on each side), so busy and
 *  smooth textures are judged alike. */
export function seamScores(
	data: ArrayLike<number>,
	width: number,
	height: number
): { x: number; y: number } {
	const at = (x: number, y: number) =>
		luma(data, ((((y % height) + height) % height) * width + (((x % width) + width) % width)) * 4);
	const near = [-3, -2, 0, 1]; // steps between neighbours just before and after the wrap
	let wrapX = 0,
		nearX = 0,
		wrapY = 0,
		nearY = 0;
	for (let y = 0; y < height; y++) {
		wrapX += Math.abs(at(0, y) - at(width - 1, y));
		for (const d of near) nearX += Math.abs(at(d + 1, y) - at(d, y));
	}
	for (let x = 0; x < width; x++) {
		wrapY += Math.abs(at(x, 0) - at(x, height - 1));
		for (const d of near) nearY += Math.abs(at(x, d + 1) - at(x, d));
	}
	const ratio = (wrap: number, nearby: number) => {
		const step = nearby / near.length;
		return step > 1e-6 ? wrap / step : wrap > 1e-6 ? Infinity : 1;
	};
	return { x: ratio(wrapX, nearX), y: ratio(wrapY, nearY) };
}

export function seamVerdict(score: number): string {
	if (score <= 1.3) return 'no visible seam';
	if (score <= 2) return 'faint seam';
	return 'visible seam';
}
