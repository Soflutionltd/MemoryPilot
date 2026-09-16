//! Lean tokenizers for the two default ONNX models.
//!
//! The `tokenizers` crate is correct but memory-hungry: loading the
//! jina/EuroBERT byte-level BPE (128k vocab, 280k merges) costs ~205 MB
//! resident and the XLM-R Unigram model used by mmarco (250k pieces)
//! ~330 MB, because its Unigram trie is a `HashMap` per node. That is
//! more than both ONNX graphs together. Both tokenizers below reproduce
//! the `tokenizers` output token-for-token (asserted in tests on FR/EN/
//! code/CJK/emoji samples) for ~42 MB and ~25 MB respectively.
//!
//! Both read the model's own `tokenizer.json`, so a new revision of the
//! vocabulary is picked up without a code change.

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use unicode_normalization::UnicodeNormalization;

// ─── Byte-level BPE (jina-embeddings-v5 / EuroBERT / Llama-3 family) ───

/// GPT-2 `bytes_to_unicode` table, inverted: the printable stand-in
/// character used in `tokenizer.json` → the raw byte it represents.
fn byte_level_decoder() -> HashMap<char, u8> {
    let mut bytes: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars: Vec<u32> = bytes.clone();
    let mut next = 0;
    for byte in 0..256u32 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            chars.push(256 + next);
            next += 1;
        }
    }
    bytes
        .iter()
        .zip(chars.iter())
        .filter_map(|(&byte, &ch)| char::from_u32(ch).map(|c| (c, byte as u8)))
        .collect()
}

/// Byte-level BPE driven by tiktoken's rank-based merge algorithm.
///
/// Valid for vocabularies exported from tiktoken-style tokenizers
/// (`ignore_merges: true`, ranks = ids), which is what EuroBERT — and
/// therefore jina-embeddings-v5 — ships. The regex pre-tokenizer pattern
/// is taken from the file, not hard-coded.
pub struct ByteLevelBpe {
    bpe: tiktoken_rs::CoreBPE,
    eos_id: u32,
    pad_id: u32,
}

impl ByteLevelBpe {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        #[derive(serde::Deserialize)]
        struct File {
            model: Model,
            added_tokens: Vec<Added>,
            pre_tokenizer: serde_json::Value,
            padding: Option<Padding>,
        }
        #[derive(serde::Deserialize)]
        struct Model {
            vocab: HashMap<String, u32>,
        }
        #[derive(serde::Deserialize)]
        struct Added {
            content: String,
            id: u32,
        }
        #[derive(serde::Deserialize)]
        struct Padding {
            pad_id: u32,
        }

        let reader = BufReader::new(
            std::fs::File::open(path)
                .map_err(|error| format!("open {}: {}", path.display(), error))?,
        );
        let file: File = serde_json::from_reader(reader)
            .map_err(|error| format!("parse {}: {}", path.display(), error))?;

        let table = byte_level_decoder();
        let mut encoder: rustc_hash::FxHashMap<Vec<u8>, u32> = Default::default();
        encoder.reserve(file.model.vocab.len());
        for (token, id) in file.model.vocab {
            let mut bytes = Vec::with_capacity(token.len());
            for ch in token.chars() {
                let byte = table
                    .get(&ch)
                    .ok_or_else(|| format!("vocab token {:?} is not byte-level encoded", token))?;
                bytes.push(*byte);
            }
            encoder.insert(bytes, id);
        }

        let mut specials: rustc_hash::FxHashMap<String, u32> = Default::default();
        let mut eos_id = None;
        for added in file.added_tokens {
            if added.content == "<|end_of_text|>" {
                eos_id = Some(added.id);
            }
            specials.insert(added.content, added.id);
        }
        let eos_id = eos_id.ok_or("tokenizer.json has no <|end_of_text|> token")?;
        let pad_id = file
            .padding
            .map(|padding| padding.pad_id)
            .ok_or("tokenizer.json has no padding.pad_id")?;

        // pre_tokenizer.pretokenizers[0] is the `Split { Regex }` step;
        // the `ByteLevel` step that follows is what the byte table above
        // undoes.
        let pattern = file.pre_tokenizer["pretokenizers"][0]["pattern"]["Regex"]
            .as_str()
            .ok_or("tokenizer.json pre_tokenizer has no Split regex")?;
        let bpe = tiktoken_rs::CoreBPE::new(encoder, specials, pattern)
            .map_err(|error| format!("BPE init: {}", error))?;
        Ok(Self { bpe, eos_id, pad_id })
    }

    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// `text` → ids, then `<|end_of_text|>`, truncated so the whole
    /// sequence is at most `max_tokens` long. The EOS token always
    /// survives truncation: it is the position the model pools.
    pub fn encode(&self, text: &str, max_tokens: usize) -> Vec<u32> {
        let mut ids = self.bpe.encode_ordinary(text);
        ids.truncate(max_tokens.saturating_sub(1));
        ids.push(self.eos_id);
        ids
    }
}

// ─── Byte-level BPE with explicit merges (GPT-2 / RoBERTa family) ───

/// GPT-2 pre-tokenizer regex. RoBERTa's `tokenizer.json` declares a
/// `ByteLevel` pre-tokenizer with `use_regex: true`, which means exactly
/// this pattern.
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// RoBERTa's byte-level BPE. Unlike the tiktoken-style vocabularies
/// above, its ids are frequency-sorted and carry no merge order, so the
/// merge list is turned into ranks (`256 + position`), tiktoken runs on
/// those ranks, and a table maps each rank back to the model id.
/// Reproduces the `tokenizers` crate token-for-token (asserted in tests).
///
/// Exposes byte offsets per token: byte-level BPE is lossless, so the
/// concatenated token bytes *are* the input, which is what lets the
/// extractive reader map a predicted span back onto the passage.
pub struct MergeBpe {
    bpe: tiktoken_rs::CoreBPE,
    rank_to_id: Vec<u32>,
    rank_bytes: Vec<Vec<u8>>,
    bos_id: u32,
    eos_id: u32,
    pad_id: u32,
}

impl MergeBpe {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        #[derive(serde::Deserialize)]
        struct File {
            model: Model,
            added_tokens: Vec<Added>,
        }
        #[derive(serde::Deserialize)]
        struct Model {
            vocab: HashMap<String, u32>,
            merges: Vec<serde_json::Value>,
        }
        #[derive(serde::Deserialize)]
        struct Added {
            content: String,
            id: u32,
        }

        let reader = BufReader::new(
            std::fs::File::open(path)
                .map_err(|error| format!("open {}: {}", path.display(), error))?,
        );
        let file: File = serde_json::from_reader(reader)
            .map_err(|error| format!("parse {}: {}", path.display(), error))?;

        let table = byte_level_decoder();
        let decode = |token: &str| -> Option<Vec<u8>> {
            token.chars().map(|ch| table.get(&ch).copied()).collect()
        };

        let rank_count = 256 + file.model.merges.len();
        let mut encoder: rustc_hash::FxHashMap<Vec<u8>, u32> = Default::default();
        encoder.reserve(rank_count);
        let mut rank_to_id = vec![u32::MAX; rank_count];
        let mut rank_bytes: Vec<Vec<u8>> = vec![Vec::new(); rank_count];

        // Ranks 0..256: the single bytes, in byte order.
        for (token, &id) in &file.model.vocab {
            if let Some(bytes) = decode(token) {
                if bytes.len() == 1 {
                    let rank = bytes[0] as usize;
                    encoder.insert(bytes.clone(), rank as u32);
                    rank_to_id[rank] = id;
                    rank_bytes[rank] = bytes;
                }
            }
        }
        // Ranks 256..: one per merge, in merge order.
        for (position, merge) in file.model.merges.iter().enumerate() {
            let merged = match merge {
                serde_json::Value::String(pair) => pair.replacen(' ', "", 1),
                serde_json::Value::Array(parts) => parts
                    .iter()
                    .filter_map(|part| part.as_str())
                    .collect::<String>(),
                _ => return Err("tokenizer.json merge entry has an unknown shape".into()),
            };
            let id = *file
                .model
                .vocab
                .get(&merged)
                .ok_or_else(|| format!("merge result {:?} missing from vocab", merged))?;
            let bytes = decode(&merged)
                .ok_or_else(|| format!("merge result {:?} is not byte-level encoded", merged))?;
            let rank = 256 + position;
            encoder.insert(bytes.clone(), rank as u32);
            rank_to_id[rank] = id;
            rank_bytes[rank] = bytes;
        }
        if rank_to_id[..256].iter().any(|&id| id == u32::MAX) {
            return Err("vocab does not cover all 256 byte tokens".into());
        }

        let special = |name: &str| -> Result<u32, String> {
            file.added_tokens
                .iter()
                .find(|added| added.content == name)
                .map(|added| added.id)
                .ok_or_else(|| format!("tokenizer.json has no {} token", name))
        };
        let bos_id = special("<s>")?;
        let eos_id = special("</s>")?;
        let pad_id = special("<pad>")?;

        let specials: rustc_hash::FxHashMap<String, u32> = Default::default();
        let bpe = tiktoken_rs::CoreBPE::new(encoder, specials, GPT2_PATTERN)
            .map_err(|error| format!("BPE init: {}", error))?;
        Ok(Self {
            bpe,
            rank_to_id,
            rank_bytes,
            bos_id,
            eos_id,
            pad_id,
        })
    }

    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    /// Model ids of `text` (no specials) with, per token, the byte range
    /// of `text` it covers.
    pub fn encode_with_offsets(&self, text: &str) -> (Vec<u32>, Vec<(usize, usize)>) {
        let ranks = self.bpe.encode_ordinary(text);
        let mut ids = Vec::with_capacity(ranks.len());
        let mut offsets = Vec::with_capacity(ranks.len());
        let mut cursor = 0usize;
        for rank in ranks {
            let width = self.rank_bytes[rank as usize].len();
            ids.push(self.rank_to_id[rank as usize]);
            offsets.push((cursor, cursor + width));
            cursor += width;
        }
        (ids, offsets)
    }

    /// Model ids of `text`, no specials.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_with_offsets(text).0
    }
}

// ─── Unigram / SentencePiece (XLM-R family: mmarco-mMiniLMv2) ───

/// Unigram language-model tokenizer (Viterbi best segmentation), with
/// the XLM-R pipeline: NFKC normalisation, whitespace split, `▁` prefix
/// per word, `<unk>` fusion, `<s> A </s></s> B </s>` pair template.
///
/// Pieces live in one `HashMap<Box<str>, (id, score)>`; the segmentation
/// probes every substring up to the longest piece length, which for
/// short memory snippets is faster than the crate's trie anyway.
pub struct Unigram {
    pieces: HashMap<Box<str>, (u32, f32)>,
    max_piece_chars: usize,
    unk_score: f32,
    unk_id: u32,
    bos_id: u32,
    eos_id: u32,
    pad_id: u32,
}

/// Penalty applied to characters absent from the vocabulary, as in
/// `tokenizers` (`K_UNK_PENALTY`).
const UNK_PENALTY: f32 = 10.0;

impl Unigram {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        #[derive(serde::Deserialize)]
        struct File {
            model: Model,
            added_tokens: Vec<Added>,
        }
        #[derive(serde::Deserialize)]
        struct Model {
            vocab: Vec<(String, f32)>,
            unk_id: u32,
        }
        #[derive(serde::Deserialize)]
        struct Added {
            content: String,
            id: u32,
        }

        let reader = BufReader::new(
            std::fs::File::open(path)
                .map_err(|error| format!("open {}: {}", path.display(), error))?,
        );
        let file: File = serde_json::from_reader(reader)
            .map_err(|error| format!("parse {}: {}", path.display(), error))?;

        let mut pieces = HashMap::with_capacity(file.model.vocab.len());
        let mut max_piece_chars = 0;
        let mut min_score = f32::INFINITY;
        for (id, (piece, score)) in file.model.vocab.into_iter().enumerate() {
            max_piece_chars = max_piece_chars.max(piece.chars().count());
            min_score = min_score.min(score);
            pieces.insert(piece.into_boxed_str(), (id as u32, score));
        }

        let special = |name: &str| -> Result<u32, String> {
            file.added_tokens
                .iter()
                .find(|added| added.content == name)
                .map(|added| added.id)
                .ok_or_else(|| format!("tokenizer.json has no {} token", name))
        };

        Ok(Self {
            pieces,
            max_piece_chars: max_piece_chars.max(1),
            unk_score: min_score - UNK_PENALTY,
            unk_id: file.model.unk_id,
            bos_id: special("<s>")?,
            eos_id: special("</s>")?,
            pad_id: special("<pad>")?,
        })
    }

    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// Best segmentation of one `▁word` into piece ids, appended to
    /// `out`. Consecutive `<unk>` are fused into one, as `tokenizers`
    /// does for this model.
    fn segment(&self, word: &str, out: &mut Vec<u32>) {
        let offsets: Vec<usize> = word
            .char_indices()
            .map(|(offset, _)| offset)
            .chain(std::iter::once(word.len()))
            .collect();
        let chars = offsets.len() - 1;
        // best[i] = (score, previous boundary, id of the piece ending at i)
        let mut best: Vec<(f32, usize, u32)> = vec![(f32::NEG_INFINITY, 0, 0); chars + 1];
        best[0].0 = 0.0;
        for start in 0..chars {
            let base = best[start].0;
            if base == f32::NEG_INFINITY {
                continue;
            }
            let mut matched = false;
            let furthest = (start + self.max_piece_chars).min(chars);
            for end in (start + 1)..=furthest {
                if let Some(&(id, score)) = self.pieces.get(&word[offsets[start]..offsets[end]]) {
                    matched = true;
                    let candidate = base + score;
                    if candidate > best[end].0 {
                        best[end] = (candidate, start, id);
                    }
                }
            }
            if !matched {
                let candidate = base + self.unk_score;
                if candidate > best[start + 1].0 {
                    best[start + 1] = (candidate, start, self.unk_id);
                }
            }
        }
        let first = out.len();
        let mut position = chars;
        while position > 0 {
            let (_, previous, id) = best[position];
            out.push(id);
            position = previous;
        }
        out[first..].reverse();
        // Fuse runs of <unk>.
        let mut write = first;
        for read in first..out.len() {
            let id = out[read];
            if id == self.unk_id && write > first && out[write - 1] == self.unk_id {
                continue;
            }
            out[write] = id;
            write += 1;
        }
        out.truncate(write);
    }

    fn encode_body(&self, text: &str, out: &mut Vec<u32>) {
        let normalized: String = text.nfkc().collect();
        let mut word = String::new();
        for token in normalized.split_whitespace() {
            word.clear();
            word.push('\u{2581}');
            word.push_str(token);
            self.segment(&word, out);
        }
    }

    /// Cross-encoder input `<s> query </s></s> document </s>`, the
    /// document truncated so the whole sequence fits in `max_tokens`.
    pub fn encode_pair(&self, query: &str, document: &str, max_tokens: usize) -> Vec<u32> {
        let mut ids = vec![self.bos_id];
        self.encode_body(query, &mut ids);
        ids.push(self.eos_id);
        ids.push(self.eos_id);
        let document_start = ids.len();
        self.encode_body(document, &mut ids);
        let document_budget = max_tokens.saturating_sub(document_start + 1);
        ids.truncate(document_start + document_budget);
        ids.push(self.eos_id);
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLES: &[&str] = &[
        "comment publier un post Instagram depuis Sociomator",
        "Décision technique : McpHub passe en transport daemon SSE avec fallback stdio, timeout réel 30 s.",
        "Règle critique de déploiement cross-projets: ne jamais déployer Docix/Konta sur Cloudflare Pages project notegenius.",
        "Projet Supabase Sociomator : project_id = nlneghlddodrleewlsfu (PAS avxxnwpujpfrmnkproow qui est Onlyst).",
        "TestFlight export compliance usesNonExemptEncryption=false, build 1.0.0+26 buildId 585f0e3a-4b45-4e0f-855b",
        "https://sociomator.com/fr/blog/how-to-plan-your-week?utm_source=app&x=1",
        "fn rerank_intra_threads() -> usize { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2) }",
        "Élève, Œuvre, ça, où, déjà, forêt, naïve, ﬁnal (ligature), ½ litre, № 5, ①",
        "日本語のテキストと中文文本、그리고 한국어. Русский текст. العربية. עברית",
        "Emoji test 🚀🔥 and  double  spaces\tand\ttabs\nnewline",
        "MAJUSCULES ET Ponctuation !!! ??? ... « guillemets » “quotes” — tiret",
        "Query: quelle est la couleur primaire du design system ?",
        "Document: [Soflution_Design_System] Couleur primaire: Fuchsia (#d946ef)",
        "12345678901234567890 3.14159 1e-9 -42 0xDEADBEEF",
        "",
        "   ",
        "a",
    ];

    fn jina_tokenizer_json() -> std::path::PathBuf {
        crate::embedding::hf_file(
            "jinaai/jina-embeddings-v5-text-nano-retrieval",
            "tokenizer.json",
            false,
        )
        .expect("jina tokenizer.json")
    }

    fn mmarco_tokenizer_json() -> std::path::PathBuf {
        crate::embedding::hf_file("SugoLabs/mmarco-mMiniLMv2-L12-H384-v1", "tokenizer.json", false)
            .expect("mmarco tokenizer.json")
    }

    #[test]
    fn byte_level_bpe_matches_tokenizers_crate() {
        let path = jina_tokenizer_json();
        let ours = ByteLevelBpe::from_file(&path).expect("load");
        let reference = tokenizers::Tokenizer::from_file(&path).expect("reference");
        for sample in SAMPLES {
            let expected = reference.encode(*sample, true).unwrap().get_ids().to_vec();
            assert_eq!(ours.encode(sample, 8192), expected, "sample {:?}", sample);
        }
    }

    #[test]
    fn byte_level_bpe_truncation_keeps_eos_last() {
        let ours = ByteLevelBpe::from_file(&jina_tokenizer_json()).expect("load");
        let long = "mémoire ".repeat(600);
        let ids = ours.encode(&long, 512);
        assert_eq!(ids.len(), 512);
        assert_eq!(*ids.last().unwrap(), ours.encode("x", 8).last().copied().unwrap());
    }

    #[test]
    fn merge_bpe_matches_tokenizers_crate_with_offsets() {
        let path = crate::embedding::hf_file(crate::reader::READER_REPO, "tokenizer.json", false)
            .expect("roberta tokenizer.json");
        let ours = MergeBpe::from_file(&path).expect("load");
        let reference = tokenizers::Tokenizer::from_file(&path).expect("reference");
        for sample in SAMPLES {
            let expected = reference.encode(*sample, false).unwrap().get_ids().to_vec();
            let (ids, offsets) = ours.encode_with_offsets(sample);
            assert_eq!(ids, expected, "sample {:?}", sample);
            // Offsets tile the input exactly.
            let mut cursor = 0;
            for (start, end) in &offsets {
                assert_eq!(*start, cursor);
                cursor = *end;
            }
            assert_eq!(cursor, sample.len());
        }
        assert_eq!(ours.bos_id(), 0);
        assert_eq!(ours.eos_id(), 2);
        assert_eq!(ours.pad_id(), 1);
    }

    #[test]
    fn unigram_matches_tokenizers_crate_on_pairs() {
        let path = mmarco_tokenizer_json();
        let ours = Unigram::from_file(&path).expect("load");
        let mut reference = tokenizers::Tokenizer::from_file(&path).expect("reference");
        reference
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 512,
                strategy: tokenizers::TruncationStrategy::OnlySecond,
                ..Default::default()
            }))
            .unwrap();
        let query = "comment publier un post Instagram";
        for sample in SAMPLES {
            let expected = reference
                .encode((query, *sample), true)
                .unwrap()
                .get_ids()
                .to_vec();
            assert_eq!(ours.encode_pair(query, sample, 512), expected, "sample {:?}", sample);
        }
        let long = "mémoire longue ".repeat(400);
        let expected = reference.encode((query, long.as_str()), true).unwrap().get_ids().to_vec();
        let ids = ours.encode_pair(query, &long, 512);
        assert_eq!(ids.len(), 512);
        assert_eq!(ids, expected);
    }
}
