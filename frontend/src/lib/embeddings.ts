// frontend/src/lib/embeddings.ts
// Textual inversion embeddings are used as in A1111: their token (the file name without
// extension) written in the prompt or negative prompt. The server loads the ones it finds;
// these helpers find, add and remove tokens with the same whole-word rule as the server.

export interface EmbeddingDetail {
	name: string;
	token: string;
	family: string | null;
	family_label: string | null;
	vectors: number;
	fits: boolean | null; // for the selected model; null when unknown
	error: string | null;
}

const EXTENSIONS = ['.safetensors', '.pt', '.bin', '.pth'];

function escape(text: string): string {
	return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** ComfyUI's `embedding:name(.ext)` written as the plain token. */
export function normalizePrompt(text: string): string {
	return text.replace(/\bembedding:([^\s,()[\]{}<>|:]+)/g, (_, word: string) => {
		const ext = EXTENSIONS.find((e) => word.toLowerCase().endsWith(e));
		return ext ? word.slice(0, -ext.length) : word;
	});
}

function tokenRegex(token: string, flags = ''): RegExp {
	return new RegExp(`(?<![\\w\\-.])${escape(token)}(?![\\w\\-])`, flags);
}

export function hasToken(text: string | null | undefined, token: string): boolean {
	return !!text && tokenRegex(token).test(normalizePrompt(text));
}

/** Append the token as one more comma-separated part. */
export function addToken(text: string | null | undefined, token: string): string {
	const current = (text ?? '').trimEnd();
	if (!current) return token;
	return current.endsWith(',') ? `${current} ${token}` : `${current}, ${token}`;
}

/** Remove every use of the token (also `(token:1.2)` and `embedding:token`). */
export function removeToken(text: string | null | undefined, token: string): string {
	const t = escape(token);
	return normalizePrompt(text ?? '')
		.replace(new RegExp(`\\(\\s*${t}\\s*(?::\\s*[\\d.]+\\s*)?\\)`, 'g'), '')
		.replace(tokenRegex(token, 'g'), '')
		.replace(/\s*,(\s*,)+/g, ',')
		.replace(/^[\s,]+|[\s,]+$/g, '')
		.replace(/ {2,}/g, ' ');
}

export interface UsedEmbedding {
	embedding: EmbeddingDetail;
	where: 'prompt' | 'negative_prompt';
}

/** Installed embeddings whose token appears in the prompt or the negative prompt. */
export function usedEmbeddings(
	prompt: string | null | undefined,
	negative: string | null | undefined,
	installed: EmbeddingDetail[]
): UsedEmbedding[] {
	const used: UsedEmbedding[] = [];
	const seen = new Set<string>();
	for (const embedding of installed) {
		if (seen.has(embedding.token)) continue; // the first file with a token wins, as on the server
		seen.add(embedding.token);
		if (hasToken(prompt, embedding.token)) used.push({ embedding, where: 'prompt' });
		if (hasToken(negative, embedding.token)) used.push({ embedding, where: 'negative_prompt' });
	}
	return used;
}
