import React, { useCallback, useEffect, useState } from "react";
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import {
  EMBEDDING_GEMMA_2_TEXT_270M,
  EmbeddingTask,
  topK,
  useEmbedder,
} from "react-native-litert-lm";
import { ProgressBar, SectionLabel } from "./ui";
import { T } from "../theme";

/** A tiny personal knowledge base — all embedded and searched on device. */
const NOTES = [
  "Dentist appointment moved to Thursday at 3pm.",
  "Sourdough: feed the starter 12 hours before mixing the dough.",
  "Flight to Tokyo departs SFO on May 4, seat 32A.",
  "The garage door code is the year we moved in.",
  "Mom's birthday dinner — book the Italian place on 5th.",
  "Quarterly taxes are due on the 15th; receipts are in the blue folder.",
  "Running plan: 5k on Tuesday, intervals on Thursday, long run Sunday.",
  "Wi-Fi password for the cabin is on the fridge magnet.",
  "Renew the passport at least six months before international travel.",
  "Tomato plants need watering every other day in July.",
  "The team offsite is in Lisbon; pack a light jacket.",
  "Return the library books before the 20th to avoid fines.",
];

/** Matryoshka: 256 of 768 dimensions is plenty for a dozen notes. */
const OPTIONS = { normalize: true, outputSize: 256 } as const;

/**
 * Semantic search with EmbeddingGemma 2 (165 MB): the notes are embedded once
 * in a single batch, each query is embedded on submit, and topK ranks the
 * notes by cosine similarity — no server, no vector database.
 */
export function SearchPanel() {
  const { isReady, downloadProgress, error, modelInfo, embed, embedBatch, load } = useEmbedder(
    EMBEDDING_GEMMA_2_TEXT_270M,
    { autoLoad: false, maxInputLength: 256 },
  );
  const [index, setIndex] = useState<Float32Array[] | null>(null);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<{ note: string; score: number }[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Embed the notes once the model is ready (the document side of search).
  useEffect(() => {
    if (!isReady || index) return;
    const started = Date.now();
    embedBatch(NOTES.map(EmbeddingTask.document), OPTIONS)
      .then((vectors) => {
        setIndex(vectors);
        setStatus(`Indexed ${NOTES.length} notes in ${Date.now() - started} ms`);
      })
      .catch((e) => setStatus(`Indexing failed: ${e.message}`));
  }, [isReady, index, embedBatch]);

  const search = useCallback(async () => {
    if (!index || !query.trim()) return;
    setBusy(true);
    try {
      const started = Date.now();
      const vector = await embed(EmbeddingTask.searchQuery(query), OPTIONS);
      setHits(topK(vector, index, 3).map(({ index: i, score }) => ({ note: NOTES[i], score })));
      setStatus(`Query embedded in ${Date.now() - started} ms`);
    } catch (e: any) {
      setStatus(`Search failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }, [index, query, embed]);

  const downloading = downloadProgress > 0 && downloadProgress < 1;

  if (!isReady) {
    return (
      <View style={s.card}>
        <Text style={s.title}>Semantic search</Text>
        <Text style={s.body}>
          EmbeddingGemma 2 turns text into vectors on this device. Load it (165 MB, once) to
          search {NOTES.length} sample notes by meaning rather than keywords.
        </Text>
        {downloading ? (
          <ProgressBar fraction={downloadProgress} />
        ) : (
          <TouchableOpacity testID="load-embedder" style={s.button} onPress={load}>
            <Text style={s.buttonText}>
              {downloadProgress === 1 && !error ? "Loading…" : "Load embedding model"}
            </Text>
          </TouchableOpacity>
        )}
        {error ? <Text style={s.error}>{error}</Text> : null}
      </View>
    );
  }

  return (
    <View style={{ gap: 14 }}>
      <View style={s.card}>
        <Text style={s.title}>Ask your notes</Text>
        <Text style={s.body}>
          {modelInfo?.embedding?.dimension ?? 768}-dim vectors, truncated to {OPTIONS.outputSize}.
          Try “teeth cleaning” or “what do I need for the garden?”
        </Text>
        <View style={s.searchRow}>
          <TextInput
            testID="search-input"
            style={s.input}
            value={query}
            onChangeText={setQuery}
            onSubmitEditing={search}
            placeholder="Search by meaning…"
            placeholderTextColor={T.dim}
            returnKeyType="search"
            editable={!!index}
          />
          <TouchableOpacity
            testID="search-btn"
            style={[s.go, (!index || busy) && { opacity: 0.4 }]}
            disabled={!index || busy}
            onPress={search}
          >
            {busy ? <ActivityIndicator color={T.bg} /> : <Text style={s.goText}>⌕</Text>}
          </TouchableOpacity>
        </View>
        {status ? <Text style={s.meta}>{status}</Text> : null}
      </View>

      {hits.length > 0 && (
        <View>
          <SectionLabel>Closest notes</SectionLabel>
          {hits.map((hit) => (
            <View key={hit.note} style={s.hit}>
              <Text style={s.hitText}>{hit.note}</Text>
              <Text style={s.score}>{hit.score.toFixed(3)}</Text>
            </View>
          ))}
        </View>
      )}
    </View>
  );
}

const s = StyleSheet.create({
  card: { backgroundColor: T.card, borderRadius: 16, padding: 16, gap: 10 },
  title: { color: T.text, fontSize: 17, fontWeight: "600" },
  body: { color: T.dim, fontSize: 14, lineHeight: 20 },
  button: {
    backgroundColor: T.accent,
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: "center",
  },
  buttonText: { color: T.text, fontSize: 15, fontWeight: "600" },
  error: { color: T.error, fontSize: 13 },
  searchRow: { flexDirection: "row", gap: 8, alignItems: "center" },
  input: {
    flex: 1,
    backgroundColor: T.elevated,
    color: T.text,
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 15,
  },
  go: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: T.sendActive,
    alignItems: "center",
    justifyContent: "center",
  },
  goText: { color: T.bg, fontSize: 20, fontWeight: "700" },
  meta: { color: T.dim, fontSize: 12 },
  hit: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: T.surface,
    borderRadius: 12,
    padding: 12,
    marginTop: 8,
    gap: 10,
  },
  hitText: { color: T.text, fontSize: 14, flex: 1, lineHeight: 19 },
  score: { color: T.cyan, fontSize: 13, fontVariant: ["tabular-nums"] },
});
