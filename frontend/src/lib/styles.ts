// frontend/src/lib/styles.ts
// Prompt styles (A1111 styles.csv): the list from the server, grouped for the picker, and
// the preview of the prompts a render will use.

export type StylePosition = 'template' | 'before' | 'after' | 'chunk';

export interface PromptStyle {
	name: string;
	group: string;
	prompt: string;
	negative_prompt: string;
	file: string;
}

export interface StylesList {
	folder: string;
	files: { name: string; styles: number; error: string | null }[];
	styles: PromptStyle[];
}

export interface TokenCount {
	tokens: number;
	chunks: number;
	limit: number;
}

export interface StylePreview {
	prompt: string;
	negative_prompt: string | null;
	loras: { name: string; weight: number }[];
	notes: Record<string, unknown>;
	model_family: string | null;
	reads_syntax: boolean;
	tokens: { prompt: TokenCount | null; negative_prompt: TokenCount | null } | null;
}

export const POSITIONS: { value: StylePosition; label: string }[] = [
	{ value: 'template', label: 'Template ({prompt}, A1111)' },
	{ value: 'before', label: 'Before the prompt' },
	{ value: 'after', label: 'After the prompt' },
	{ value: 'chunk', label: 'Own chunk (BREAK)' }
];

/** Styles matching the search (name or text), grouped by the part of the name before ":". */
export function groupStyles(styles: PromptStyle[], search = ''): [string, PromptStyle[]][] {
	const q = search.trim().toLowerCase();
	const groups = new Map<string, PromptStyle[]>();
	for (const style of styles) {
		if (
			q &&
			!style.name.toLowerCase().includes(q) &&
			!style.prompt.toLowerCase().includes(q) &&
			!style.negative_prompt.toLowerCase().includes(q)
		)
			continue;
		const list = groups.get(style.group) ?? [];
		list.push(style);
		groups.set(style.group, list);
	}
	return [...groups.entries()];
}

/** "92 tokens · 2 chunks", with a warning when plain text would cut it. */
export function tokenSummary(count: TokenCount | null | undefined, plain: boolean): string {
	if (!count) return '';
	const base = `${count.tokens} token${count.tokens === 1 ? '' : 's'}`;
	if (plain) return count.tokens > count.limit ? `${base}: cut at ${count.limit}` : base;
	return count.chunks > 1 ? `${base} · ${count.chunks} chunks of ${count.limit}` : base;
}

/** A preview is worth asking for: styles chosen, or wildcards / LoRA tags in the prompts. */
export function needsPreview(
	prompt: string | null | undefined,
	negative: string | null | undefined,
	styles: string[] | null | undefined
): boolean {
	if (styles?.length) return true;
	const text = `${prompt ?? ''} ${negative ?? ''}`;
	return /<(lora|lyco|hypernet):/i.test(text) || /\{[^{}]*\|[^{}]*\}/.test(text);
}
