// frontend/src/lib/tokens.ts
// The token view of a prompt (POST /prompt/tokens): chunks of up to 75 tokens with weights.

export interface TokenPiece {
	t: string; // the token's text
	w: number; // its weight (1 = normal)
	end: boolean; // the token ends a word
	emb: boolean; // a textual inversion embedding vector
}

export interface TokenView {
	tokens: number;
	limit: number;
	chunks: TokenPiece[][];
	dropped: TokenPiece[]; // plain text only: cut off after `limit`
	edited: boolean;
}

/** "92 · 2 chunks", or for plain text "92: 17 cut off". */
export function tokenLine(v: TokenView, plain: boolean): string {
	if (plain && v.dropped.length) return `${v.tokens}: ${v.dropped.length} cut off`;
	return v.chunks.length > 1 ? `${v.tokens} · ${v.chunks.length} chunks` : `${v.tokens}`;
}

/** A background that shows the weight: indigo above 1, faded below 1. */
export function weightStyle(w: number): string {
	if (w > 1)
		return `background-color: rgb(99 102 241 / ${Math.min(0.55, (w - 1) * 0.9).toFixed(2)})`;
	if (w < 1) return `opacity: ${Math.max(0.35, w).toFixed(2)}`;
	return '';
}
