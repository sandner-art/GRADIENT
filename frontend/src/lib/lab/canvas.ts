// frontend/src/lib/lab/canvas.ts
// Browser-only pixel helpers.

/** Exact pixels of a base64 image (no color management, no premultiplication). */
export async function decodePixels(dataUrl: string): Promise<ImageData> {
	const blob = await (await fetch(dataUrl)).blob();
	const bitmap = await createImageBitmap(blob, {
		colorSpaceConversion: 'none',
		premultiplyAlpha: 'none'
	});
	const canvas = document.createElement('canvas');
	canvas.width = bitmap.width;
	canvas.height = bitmap.height;
	const ctx = canvas.getContext('2d', { willReadFrequently: true });
	if (!ctx) throw new Error('Canvas 2D is not available in this browser');
	ctx.drawImage(bitmap, 0, 0);
	bitmap.close();
	return ctx.getImageData(0, 0, canvas.width, canvas.height);
}
