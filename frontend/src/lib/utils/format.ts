// frontend/src/lib/utils/format.ts

/** 1536 -> "1.5 KB" (binary units, like file managers). */
export function formatBytes(bytes: number | null | undefined, digits = 1): string {
	if (bytes === null || bytes === undefined || !Number.isFinite(bytes) || bytes < 0) return '?';
	const units = ['B', 'KB', 'MB', 'GB', 'TB'];
	let value = bytes;
	let unit = 0;
	while (value >= 1024 && unit < units.length - 1) {
		value /= 1024;
		unit++;
	}
	return unit === 0 ? `${Math.round(value)} B` : `${value.toFixed(digits)} ${units[unit]}`;
}

/** Seconds -> "1h 05m", "3m 20s", "12s". */
export function formatDuration(seconds: number | null | undefined): string {
	if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0)
		return '';
	const s = Math.round(seconds);
	if (s < 60) return `${s}s`;
	const m = Math.floor(s / 60);
	if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
	return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

/** Remaining time for a download, or '' when unknown. */
export function etaSeconds(total?: number | null, done?: number, speed?: number): number | null {
	if (!total || !speed || speed <= 0 || done === undefined) return null;
	return Math.max(0, (total - done) / speed);
}

/** A scheduler default like "EulerDiscreteScheduler(timestep_spacing=trailing)" is only usable if it is a plain class name. */
export function plainSchedulerName(value?: string | null): string | null {
	return value && /^[A-Za-z0-9_]+$/.test(value) ? value : null;
}

/** Readable message for an error thrown by apiClient (FastAPI detail string or validation list). */
export function apiErrorMessage(e: unknown): string {
	if (typeof e === 'object' && e !== null && 'status' in e) {
		const { status, detail } = e as { status: number; detail?: unknown };
		if (typeof detail === 'string') return status > 0 ? `Error ${status}: ${detail}` : detail;
		if (Array.isArray(detail))
			return `Invalid request: ${detail
				.map((d: { loc?: unknown[]; msg?: string }) => {
					const where = (d.loc ?? []).filter((p) => p !== 'body').join('.');
					return where ? `${where}: ${d.msg}` : d.msg;
				})
				.join('; ')}`;
		if (detail) return `Error ${status}: ${JSON.stringify(detail)}`;
		return `Error ${status}`;
	}
	return e instanceof Error ? e.message : String(e);
}
