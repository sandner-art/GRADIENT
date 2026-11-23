// frontend/src/lib/stores/queue.svelte.ts
// The render queue as seen by the UI: live updates over server-sent events, plus actions.

import { del, get, post } from '#lib/apiClient.js';
import type { JobKind, QueueEvent, QueueJob } from '#lib/types.js';

export const queue = $state<{
	jobs: Record<string, QueueJob>;
	order: string[];
	connected: boolean;
}>({ jobs: {}, order: [], connected: false });

type Listener = (job: QueueJob, previous: QueueJob | undefined) => void;
// callback registry, not UI state
// eslint-disable-next-line svelte/prefer-svelte-reactivity
const listeners = new Set<Listener>();

/** Called whenever a job changes (used for "images ready" notices). */
export function onJobChange(listener: Listener): () => void {
	listeners.add(listener);
	return () => listeners.delete(listener);
}

function put(job: QueueJob) {
	const previous = queue.jobs[job.id];
	queue.jobs[job.id] = job;
	if (job.status === 'queued' && !queue.order.includes(job.id)) queue.order.push(job.id);
	if (job.status !== 'queued') queue.order = queue.order.filter((id) => id !== job.id);
	for (const listener of listeners) listener(job, previous ? { ...previous } : undefined);
}

export function applyEvent(event: QueueEvent) {
	if (event.type === 'snapshot') {
		queue.jobs = Object.fromEntries(event.jobs.map((j) => [j.id, j]));
		queue.order = event.jobs
			.filter((j) => j.status === 'queued')
			.sort((a, b) => (a.position ?? 0) - (b.position ?? 0))
			.map((j) => j.id);
	} else if (event.type === 'job') {
		put(event.job);
	} else if (event.type === 'removed') {
		for (const id of event.ids) delete queue.jobs[id];
	} else if (event.type === 'order') {
		queue.order = event.order.filter((id) => queue.jobs[id]);
	}
}

let source: EventSource | null = null;
let poll: ReturnType<typeof setInterval> | null = null;

/** Connect once (idempotent); falls back to polling when EventSource is unavailable. */
export function connectQueue() {
	if (typeof window === 'undefined' || source || poll) return;
	if (typeof EventSource === 'undefined') {
		const refresh = async () => {
			try {
				const data = await get<{ jobs: QueueJob[] }>('/queue');
				applyEvent({ type: 'snapshot', jobs: data.jobs });
				queue.connected = true;
			} catch {
				queue.connected = false;
			}
		};
		refresh();
		poll = setInterval(refresh, 1500);
		return;
	}
	source = new EventSource('/api/v1/queue/events');
	source.onopen = () => (queue.connected = true);
	source.onerror = () => (queue.connected = false); // EventSource reconnects by itself
	source.onmessage = (message) => {
		try {
			applyEvent(JSON.parse(message.data) as QueueEvent);
		} catch (e) {
			console.error('Bad queue event', e);
		}
	};
}

export function disconnectQueue() {
	source?.close();
	source = null;
	if (poll) clearInterval(poll);
	poll = null;
}

// ---- views
const FINISHED = ['completed', 'failed', 'cancelled'];

export function runningJob(): QueueJob | null {
	return Object.values(queue.jobs).find((j) => j.status === 'running') ?? null;
}

export function waitingJobs(): QueueJob[] {
	return queue.order.map((id) => queue.jobs[id]).filter(Boolean);
}

export function finishedJobs(limit = 50): QueueJob[] {
	return Object.values(queue.jobs)
		.filter((j) => FINISHED.includes(j.status))
		.sort((a, b) => (b.finished ?? b.created) - (a.finished ?? a.created))
		.slice(0, limit);
}

/** 0..1 progress of a running job (grids count finished cells). */
export function jobFraction(job: QueueJob): number | null {
	const p = job.progress ?? {};
	const step = p.total ? Math.min(1, (p.step ?? 0) / p.total) : null;
	if (p.items && p.items > 1) return Math.min(1, ((p.item ?? 1) - 1 + (step ?? 0)) / p.items);
	return step;
}

export function outputIds(job: QueueJob | undefined): number[] {
	return (job?.result?.outputs ?? []).map((o) => o.id).filter((id): id is number => id !== null);
}

// ---- actions
export async function submitJob(
	kind: Extract<JobKind, 'text_to_image' | 'image_to_image' | 'xyz_grid'>,
	payload: Record<string, unknown>,
	options: { count?: number; label?: string } = {}
): Promise<QueueJob[]> {
	const response = await post<{ jobs: QueueJob[] }>('/queue', {
		kind,
		payload,
		count: options.count ?? 1,
		label: options.label ?? null
	});
	for (const job of response.jobs) put(job);
	return response.jobs;
}

export const cancelJob = (id: string) => post<QueueJob>(`/queue/${id}/cancel`, {});
export const retryJob = (id: string) => post<QueueJob>(`/queue/${id}/retry`, {});
export const moveJob = (id: string, to: 'top' | 'up' | 'down' | 'bottom') =>
	post<{ order: string[] }>(`/queue/${id}/move`, { to });
export async function clearFinished() {
	await del<{ removed: number }>('/queue/finished');
}

/** Jobs queued from each render form, newest first (results shown under the form). */
export const formJobs = $state<{ text: string[]; image: string[] }>({ text: [], image: [] });

export function rememberFormJobs(form: 'text' | 'image', jobs: QueueJob[]) {
	formJobs[form] = [...jobs.map((j) => j.id).reverse(), ...formJobs[form]].slice(0, 40);
}
