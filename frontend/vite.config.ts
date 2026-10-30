import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';
import { storybookTest } from '@storybook/addon-vitest/vitest-plugin';
import { playwright } from '@vitest/browser-playwright';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const dirname = path.dirname(fileURLToPath(import.meta.url));

// The UI is a single-page app. `npm run build` writes it to ./build, which the
// GRADIENT API server serves at http://127.0.0.1:8000 (one process, one URL).
// During development, `npm run dev` serves it on :5173 and proxies /api to the backend.
const backend = process.env.GRADIENT_API_URL ?? 'http://127.0.0.1:8000';

export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit({
			preprocess: vitePreprocess(),
			adapter: adapter({ fallback: 'index.html', strict: false })
		})
	],
	server: {
		proxy: {
			'/api': { target: backend, changeOrigin: true },
			'/ws': { target: backend.replace(/^http/, 'ws'), ws: true }
		}
	},
	test: {
		projects: [
			{
				extends: true,
				test: {
					name: 'unit',
					environment: 'node',
					include: ['src/**/*.test.ts']
				}
			},
			{
				// Runs every story as a browser test (`npm run test:storybook`)
				extends: true,
				plugins: [storybookTest({ configDir: path.join(dirname, '.storybook') })],
				test: {
					name: 'storybook',
					browser: {
						enabled: true,
						headless: true,
						provider: playwright(),
						instances: [{ browser: 'chromium' }]
					}
				}
			}
		]
	}
});
