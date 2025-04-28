import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
    server: { // Add server configuration
        proxy: {
            // Proxy requests starting with /api to the backend server
            '/api': {
                target: 'http://127.0.0.1:8000', // Your FastAPI backend URL
                changeOrigin: true, // Recommended for virtual hosted sites
                // Optional: Rewrite path if needed, but likely not if backend uses /api prefix
                // rewrite: (path) => path.replace(/^\/api/, '')
            }
        }
    }
});
