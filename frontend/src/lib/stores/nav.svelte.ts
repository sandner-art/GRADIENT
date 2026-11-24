// frontend/src/lib/stores/nav.svelte.ts
// Browser history for in-app navigation, so Back (the mouse's back button, Alt+Left, the
// browser button) goes back inside GRADIENT: it closes the image viewer or the queue drawer,
// returns to the previous tab or gallery view. Forward works too, and the tab is in the URL
// (#gallery), so a reload keeps it.

import { pushState, replaceState } from '$app/navigation';
import { resolve } from '$app/paths';
import { page } from '$app/state';
import { untrack } from 'svelte';
import { closeMenu, ui, type GalleryView, type Tab } from './ui.svelte.js';

export interface NavState {
	tab: Tab;
	view: GalleryView;
	viewer: { ids: number[]; index: number } | null;
	queue: boolean;
}

const TABS: Tab[] = ['t2i', 'i2i', 'gallery', 'grid', 'lab', 'chat', 'models'];

export function snapshot(): NavState {
	return {
		tab: ui.tab,
		view: ui.galleryView,
		viewer: ui.viewer ? { ids: [...ui.viewer.ids], index: ui.viewer.index } : null,
		queue: ui.queueOpen
	};
}

/** Same place in the app (paging inside the viewer doesn't make a new history entry). */
export function samePlace(a: NavState | undefined, b: NavState): boolean {
	return (
		!!a &&
		a.tab === b.tab &&
		a.view === b.view &&
		a.queue === b.queue &&
		(a.viewer?.ids.join(',') ?? '') === (b.viewer?.ids.join(',') ?? '')
	);
}

/** Overlays open in `a` that are closed in `b` (closing one should consume its entry). */
export function closedOverlay(a: NavState | undefined, b: NavState): boolean {
	return !!a && ((!!a.viewer && !b.viewer) || (a.queue && !b.queue));
}

export function tabFromHash(hash: string): Tab | null {
	const t = hash.replace(/^#/, '') as Tab;
	return TABS.includes(t) ? t : null;
}

function apply(s: NavState) {
	closeMenu();
	ui.tab = s.tab;
	ui.galleryView = s.view;
	ui.queueOpen = s.queue;
	ui.viewer = s.viewer ? { ids: [...s.viewer.ids], index: s.viewer.index } : null;
}

const current = () => (page.state as { nav?: NavState }).nav;

/** Start syncing; call once after the app has mounted. Returns a cleanup function. */
export function setupNavigation(): () => void {
	const fromHash = tabFromHash(location.hash);
	if (fromHash) ui.tab = fromHash;
	// shallow routing on this page: only the #tab changes (resolve('/') keeps the base path)
	// eslint-disable-next-line svelte/no-navigation-without-resolve
	replaceState(`${resolve('/')}#${ui.tab}`, { nav: snapshot() });

	let pendingBack = false;
	let timer: ReturnType<typeof setTimeout> | undefined;

	function sync() {
		const s = snapshot();
		const cur = current();
		if (samePlace(cur, s)) return;
		if (closedOverlay(cur, s) && !pendingBack) {
			// the viewer/queue was closed in the UI: step back over its entry, so Back later
			// doesn't reopen it; any other change made at the same time is pushed after that
			pendingBack = true;
			history.back();
			return;
		}
		// eslint-disable-next-line svelte/no-navigation-without-resolve
		pushState(`${resolve('/')}#${s.tab}`, { nav: s });
	}

	return $effect.root(() => {
		// app → history (changes in one tick become one entry)
		$effect(() => {
			snapshot();
			clearTimeout(timer);
			timer = setTimeout(() => untrack(sync), 0);
		});
		// history → app (Back / Forward)
		$effect(() => {
			const target = current();
			untrack(() => {
				if (!target) return;
				if (pendingBack) {
					pendingBack = false;
					const s = snapshot();
					// eslint-disable-next-line svelte/no-navigation-without-resolve
					if (!samePlace(target, s)) pushState(`${resolve('/')}#${s.tab}`, { nav: s });
					return;
				}
				if (!samePlace(target, snapshot())) apply(target);
			});
		});
		return () => clearTimeout(timer);
	});
}
