// frontend/src/app.d.ts (or ambient.d.ts / svelte.d.ts)

/// <reference types="@sveltejs/kit" />

// Declare that modules ending in '.svelte' export a Svelte component constructor
declare module '*.svelte' {
	import type { ComponentType } from 'svelte';
	const component: ComponentType;
	export default component;
  }
  
  // You can also add other global types needed by your app here
  // For example, importing and re-exporting your API types might be useful
  // export * from '#lib/types.js'; // Assuming you create a types.ts file
