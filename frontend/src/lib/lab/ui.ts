// frontend/src/lib/lab/ui.ts
// Shared Tailwind class strings for the Lab components (same look as the other forms).

export const card =
	'rounded-lg border border-gray-200 bg-white p-5 shadow-md dark:border-gray-700 dark:bg-gray-800';
export const label = 'mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300';
export const input =
	'w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 dark:text-white';
export const smallInput =
	'w-full rounded-md border border-gray-300 px-2 py-1 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white';
export const help = 'mt-1 text-xs text-gray-500 dark:text-gray-400';
export const primaryButton =
	'inline-flex items-center justify-center rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50';
export const secondaryButton =
	'inline-flex items-center justify-center rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600';
export const errorBox =
	'rounded border border-red-400 bg-red-100 p-3 text-sm text-red-700 dark:border-red-700 dark:bg-red-900 dark:text-red-200';
export const noteBox =
	'rounded border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200';
export const heading = 'text-lg font-semibold text-gray-900 dark:text-gray-100';
export const muted = 'text-sm text-gray-500 dark:text-gray-400';

export const png = (b64: string) => `data:image/png;base64,${b64}`;
/** Lab images may be PNG or WEBP; browsers sniff the bytes, the declared type only has to be an image. */
export const img = (b64: string) =>
	b64.startsWith('UklGR') ? `data:image/webp;base64,${b64}` : png(b64);
