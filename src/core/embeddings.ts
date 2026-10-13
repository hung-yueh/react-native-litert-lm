/**
 * Embedding helpers — pure, dependency-free and react-native-free, so they run
 * the same in an app, a Node script, or a test.
 *
 * Vectors are `Float32Array`s. Those returned by `embed()` are views over
 * native-owned memory; nothing here copies them except {@link normalize}.
 */

/**
 * Task-prefix builders for EmbeddingGemma 2, matching the prompts the model was
 * trained and benchmarked with (its model card and `config_sentence_transformers.json`:
 * https://huggingface.co/google/embeddinggemma-2). LiteRT-LM's embedding page
 * shows an older `task: search query | text:` form; the model card wins.
 *
 * - **Retrieval is asymmetric**: the query gets its task prefix
 *   ({@link EmbeddingTask.searchQuery}, `questionAnswering`, `factChecking`,
 *   `codeRetrieval`), and every corpus item gets {@link EmbeddingTask.document}.
 * - **Classification, clustering and similarity are symmetric**: every input
 *   being compared gets the same prefix.
 * - **Images and audio take no prefix**; pass them as plain parts.
 *
 * Input is trimmed first. Each builder takes one argument, so it can be passed
 * straight to `Array.map`.
 *
 * @example
 * ```ts
 * const q = await embedder.embed(EmbeddingTask.searchQuery(userInput), opts);
 * const docs = await embedder.embedBatch(notes.map(EmbeddingTask.document), opts);
 * ```
 */
export const EmbeddingTask = {
  /** The query side of search (`task: search result | query: …`). */
  searchQuery: (query: string): string => `task: search result | query: ${query.trim()}`,
  /** The question side of question answering; passages use {@link EmbeddingTask.document}. */
  questionAnswering: (question: string): string =>
    `task: question answering | query: ${question.trim()}`,
  /** The claim side of fact checking; evidence uses {@link EmbeddingTask.document}. */
  factChecking: (claim: string): string => `task: fact checking | query: ${claim.trim()}`,
  /** The query side of code search; code uses {@link EmbeddingTask.document} with the file name as its title. */
  codeRetrieval: (query: string): string => `task: code retrieval | query: ${query.trim()}`,
  /**
   * A corpus item for any retrieval task (`title: … | text: …`). Pass a string,
   * or `{ title, text }` when the item has a real title; a missing title is
   * written as `none`, as the model expects.
   */
  document: (doc: string | EmbeddingDocument): string => {
    const { title, text } = typeof doc === "string" ? { title: undefined, text: doc } : doc;
    return `title: ${title?.trim() || "none"} | text: ${text.trim()}`;
  },
  /** Every input to a classifier built on the embeddings. */
  classification: (text: string): string => `task: classification | query: ${text.trim()}`,
  /** Every input to clustering. */
  clustering: (text: string): string => `task: clustering | query: ${text.trim()}`,
  /** Both sides of a similarity comparison. */
  sentenceSimilarity: (text: string): string =>
    `task: sentence similarity | query: ${text.trim()}`,
} as const;

/** A titled corpus item for {@link EmbeddingTask.document}. */
export interface EmbeddingDocument {
  /** The item's real title (a heading, subject line or file name), if it has one */
  title?: string;
  text: string;
}

function assertSameLength(a: ArrayLike<number>, b: ArrayLike<number>): void {
  if (a.length !== b.length) {
    throw new RangeError(
      `Embedding dimensions differ (${a.length} vs ${b.length}). ` +
        "Compare vectors produced with the same model and outputSize.",
    );
  }
}

/** Dot product. Equals cosine similarity when both vectors are unit length (`normalize: true`). */
export function dotProduct(a: ArrayLike<number>, b: ArrayLike<number>): number {
  assertSameLength(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/** Cosine similarity in [-1, 1]. Returns 0 if either vector is all zeros. */
export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  assertSameLength(a, b);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / Math.sqrt(na * nb);
}

/** A unit-length copy of `v` (all-zero input returns zeros). */
export function normalize(v: ArrayLike<number>): Float32Array {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  const out = new Float32Array(v.length);
  if (norm === 0) return out;
  const inv = 1 / Math.sqrt(norm);
  for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
  return out;
}

/** One ranked hit from {@link topK}. */
export interface EmbeddingMatch {
  /** Index into the `candidates` array */
  index: number;
  /** Cosine similarity to the query */
  score: number;
}

/**
 * The `k` candidates most similar to `query`, best first. A linear scan —
 * plenty for the thousands of vectors an on-device index typically holds.
 */
export function topK(
  query: ArrayLike<number>,
  candidates: ReadonlyArray<ArrayLike<number>>,
  k: number,
): EmbeddingMatch[] {
  const scored = candidates.map((c, index) => ({ index, score: cosineSimilarity(query, c) }));
  scored.sort((x, y) => y.score - x.score);
  return scored.slice(0, Math.max(0, k));
}
