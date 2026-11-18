//! Lean tokenizers for the ONNX models.
//!
//! The `tokenizers` crate is correct but memory-hungry: loading the
//! jina/EuroBERT byte-level BPE (128k vocab, 280k merges) costs ~205 MB
//! resident, more than the ONNX graph it feeds. The tokenizers below
//! reproduce the `tokenizers` output token-for-token (asserted in tests
//! on FR/EN/code/CJK/emoji samples) for a fraction of that.
//!
//! Each reads the model's own `tokenizer.json`.

use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

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

/// A merge-list BPE vocabulary turned into tiktoken ranks: the 256 single
/// bytes first, then one rank per merge in merge order, with the tables
/// that map a rank back to the model id and to its bytes.
struct MergeRanks {
    bpe: tiktoken_rs::CoreBPE,
    rank_to_id: Vec<u32>,
    rank_bytes: Vec<Vec<u8>>,
}

impl MergeRanks {
    fn build(vocab: &HashMap<String, u32>, merges: &[serde_json::Value]) -> Result<Self, String> {
        let table = byte_level_decoder();
        let decode = |token: &str| -> Option<Vec<u8>> {
            token.chars().map(|ch| table.get(&ch).copied()).collect()
        };

        let rank_count = 256 + merges.len();
        let mut encoder: rustc_hash::FxHashMap<Vec<u8>, u32> = Default::default();
        encoder.reserve(rank_count);
        let mut rank_to_id = vec![u32::MAX; rank_count];
        let mut rank_bytes: Vec<Vec<u8>> = vec![Vec::new(); rank_count];

        // Ranks 0..256: the single bytes, in byte order.
        for (token, &id) in vocab {
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
        for (position, merge) in merges.iter().enumerate() {
            let merged = match merge {
                serde_json::Value::String(pair) => pair.replacen(' ', "", 1),
                serde_json::Value::Array(parts) => parts
                    .iter()
                    .filter_map(|part| part.as_str())
                    .collect::<String>(),
                _ => return Err("tokenizer.json merge entry has an unknown shape".into()),
            };
            let id = *vocab
                .get(&merged)
                .ok_or_else(|| format!("merge result {:?} missing from vocab", merged))?;
            let bytes = decode(&merged)
                .ok_or_else(|| format!("merge result {:?} is not byte-level encoded", merged))?;
            let rank = 256 + position;
            encoder.insert(bytes.clone(), rank as u32);
            rank_to_id[rank] = id;
            rank_bytes[rank] = bytes;
        }
        let specials: rustc_hash::FxHashMap<String, u32> = Default::default();
        let bpe = tiktoken_rs::CoreBPE::new(encoder, specials, GPT2_PATTERN)
            .map_err(|error| format!("BPE init: {}", error))?;
        Ok(Self {
            bpe,
            rank_to_id,
            rank_bytes,
        })
    }
}

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

        let ranks = MergeRanks::build(&file.model.vocab, &file.model.merges)?;
        if ranks.rank_to_id[..256].iter().any(|&id| id == u32::MAX) {
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

        Ok(Self {
            bpe: ranks.bpe,
            rank_to_id: ranks.rank_to_id,
            rank_bytes: ranks.rank_bytes,
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

// ─── Added tokens ───

/// One `added_tokens` entry of a `tokenizer.json`: text matched verbatim
/// before the BPE runs. `lstrip` also swallows the whitespace before it.
#[derive(serde::Deserialize)]
struct AddedToken {
    content: String,
    id: u32,
    /// Matched on the normalised text rather than on the raw one.
    normalized: bool,
    lstrip: bool,
    rstrip: bool,
    single_word: bool,
}

/// The added tokens of a tokenizer, longest content first, so the first
/// hit at a position is the longest one.
struct AddedTokens {
    tokens: Vec<AddedToken>,
}

impl AddedTokens {
    fn new(mut tokens: Vec<AddedToken>) -> Result<Self, String> {
        if let Some(unsupported) = tokens.iter().find(|token| token.rstrip || token.single_word) {
            return Err(format!(
                "added token {:?} uses rstrip or single_word, which this tokenizer does not implement",
                unsupported.content
            ));
        }
        tokens.retain(|token| !token.content.is_empty());
        tokens.sort_by_key(|token| std::cmp::Reverse(token.content.len()));
        Ok(Self { tokens })
    }

    fn id(&self, name: &str) -> Result<u32, String> {
        self.tokens
            .iter()
            .find(|token| token.content == name)
            .map(|token| token.id)
            .ok_or_else(|| format!("tokenizer.json has no {} token", name))
    }

    /// Cuts the added tokens of one class out of `text`, leftmost-longest:
    /// their ids go to `out`, the text between them to `plain`.
    fn cut(&self, text: &str, normalized: bool, out: &mut Vec<u32>, plain: impl Fn(&str, &mut Vec<u32>)) {
        let mut plain_start = 0;
        let mut position = 0;
        while position < text.len() {
            let rest = &text[position..];
            let hit = self
                .tokens
                .iter()
                .find(|token| token.normalized == normalized && rest.starts_with(token.content.as_str()));
            match hit {
                Some(token) => {
                    let mut before = &text[plain_start..position];
                    if token.lstrip {
                        before = before.trim_end();
                    }
                    plain(before, out);
                    out.push(token.id);
                    position += token.content.len();
                    plain_start = position;
                }
                None => {
                    position += rest.chars().next().map_or(1, char::len_utf8);
                }
            }
        }
        plain(&text[plain_start..], out);
    }
}

// ─── SentencePiece-style BPE (Gemma family: mmBERT) ───

const METASPACE: char = '\u{2581}';

/// Gemma's tokenizer, which mmBERT reuses: added tokens cut out of the raw
/// text, spaces turned into `▁`, a split in front of every `▁`, then a
/// character-level BPE with an explicit merge list on each piece. A
/// character absent from the vocabulary falls back to its UTF-8 bytes
/// (`<0xE2>`…). Pair template `<bos> A <eos> B <eos>`.
/// Reproduces the `tokenizers` crate token-for-token (asserted in tests).
pub struct SentencePieceBpe {
    vocab: HashMap<Box<str>, u32>,
    /// (left id, right id) → (merge rank, merged id).
    merges: rustc_hash::FxHashMap<(u32, u32), (u32, u32)>,
    /// `<0xNN>` token of each byte; the vocabulary has none for the few
    /// bytes that are tokens of their own (tab, newline).
    byte_ids: [Option<u32>; 256],
    unk_id: u32,
    added: AddedTokens,
    bos_id: u32,
    eos_id: u32,
    pad_id: u32,
}

impl SentencePieceBpe {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        #[derive(serde::Deserialize)]
        struct File {
            model: Model,
            added_tokens: Vec<AddedToken>,
        }
        #[derive(serde::Deserialize)]
        struct Model {
            vocab: HashMap<String, u32>,
            merges: Vec<(String, String)>,
            byte_fallback: bool,
        }

        let reader = BufReader::new(
            std::fs::File::open(path)
                .map_err(|error| format!("open {}: {}", path.display(), error))?,
        );
        let file: File = serde_json::from_reader(reader)
            .map_err(|error| format!("parse {}: {}", path.display(), error))?;
        if !file.model.byte_fallback {
            return Err("tokenizer.json BPE model has no byte fallback, which this tokenizer relies on".into());
        }

        let added = AddedTokens::new(file.added_tokens)?;
        let bos_id = added.id("<bos>")?;
        let eos_id = added.id("<eos>")?;
        let pad_id = added.id("<pad>")?;

        let id_of = |token: &str| -> Result<u32, String> {
            file.model
                .vocab
                .get(token)
                .copied()
                .ok_or_else(|| format!("token {:?} is missing from the vocab", token))
        };
        let unk_id = added.id("<unk>")?;
        let mut byte_ids = [None; 256];
        for (byte, slot) in byte_ids.iter_mut().enumerate() {
            *slot = file.model.vocab.get(&format!("<0x{:02X}>", byte)).copied();
        }
        let mut merges: rustc_hash::FxHashMap<(u32, u32), (u32, u32)> = Default::default();
        merges.reserve(file.model.merges.len());
        for (rank, (left, right)) in file.model.merges.iter().enumerate() {
            let merged = id_of(&format!("{}{}", left, right))?;
            // The first occurrence of a pair is the one that counts.
            merges
                .entry((id_of(left)?, id_of(right)?))
                .or_insert((rank as u32, merged));
        }
        let vocab = file
            .model
            .vocab
            .into_iter()
            .map(|(token, id)| (token.into_boxed_str(), id))
            .collect();

        Ok(Self {
            vocab,
            merges,
            byte_ids,
            unk_id,
            added,
            bos_id,
            eos_id,
            pad_id,
        })
    }

    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// BPE of one piece (`▁word`): its characters, then the merges in
    /// rank order until none applies.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        let mut symbols: Vec<u32> = Vec::with_capacity(piece.len());
        let mut buffer = [0u8; 4];
        for ch in piece.chars() {
            let text = ch.encode_utf8(&mut buffer);
            match self.vocab.get(&*text) {
                Some(&id) => symbols.push(id),
                None => {
                    let bytes: Option<Vec<u32>> = text.bytes().map(|byte| self.byte_ids[byte as usize]).collect();
                    match bytes {
                        Some(bytes) => symbols.extend(bytes),
                        // Consecutive unknowns are fused into one.
                        None if symbols.last() == Some(&self.unk_id) => {}
                        None => symbols.push(self.unk_id),
                    }
                }
            }
        }
        loop {
            let best = symbols
                .windows(2)
                .filter_map(|pair| self.merges.get(&(pair[0], pair[1])).map(|merge| (*merge, (pair[0], pair[1]))))
                .min_by_key(|((rank, _), _)| *rank);
            let Some(((_, merged), (left, right))) = best else {
                break;
            };
            let mut write = 0;
            let mut read = 0;
            while read < symbols.len() {
                if read + 1 < symbols.len() && symbols[read] == left && symbols[read + 1] == right {
                    symbols[write] = merged;
                    read += 2;
                } else {
                    symbols[write] = symbols[read];
                    read += 1;
                }
                write += 1;
            }
            symbols.truncate(write);
        }
        out.extend(symbols);
    }

    /// One stretch of text between added tokens: spaces become `▁`, a `▁`
    /// is put in front when there is none, and every `▁` starts a piece.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut normalized = String::with_capacity(text.len() + METASPACE.len_utf8());
        if !text.starts_with([' ', METASPACE]) {
            normalized.push(METASPACE);
        }
        normalized.extend(text.chars().map(|ch| if ch == ' ' { METASPACE } else { ch }));
        let mut piece_start = 0;
        for (offset, ch) in normalized.char_indices() {
            if ch == METASPACE && offset > piece_start {
                self.encode_piece(&normalized[piece_start..offset], out);
                piece_start = offset;
            }
        }
        self.encode_piece(&normalized[piece_start..], out);
    }

    fn encode_body(&self, text: &str, out: &mut Vec<u32>) {
        self.added
            .cut(text, false, out, |plain, out| self.encode_plain(plain, out));
    }

    /// Cross-encoder input `<bos> query <eos> document <eos>`, the
    /// document truncated so the whole sequence fits in `max_tokens`.
    pub fn encode_pair(&self, query: &str, document: &str, max_tokens: usize) -> Vec<u32> {
        let mut ids = vec![self.bos_id];
        self.encode_body(query, &mut ids);
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
        "Document: [MonApp_Design_System] Couleur primaire: Fuchsia (#d946ef)",
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
    fn sentencepiece_bpe_matches_tokenizers_crate_on_pairs() {
        let spec = crate::reranking::HORIZON_SMALL_INT8;
        let path = crate::model_mirror::mirrored_file(spec.mirror_tag, &spec.tokenizer).expect("reranker tokenizer.json");
        let ours = SentencePieceBpe::from_file(&path).expect("load");
        let mut reference = tokenizers::Tokenizer::from_file(&path).expect("reference");
        reference
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 512,
                strategy: tokenizers::TruncationStrategy::OnlySecond,
                ..Default::default()
            }))
            .unwrap();
        reference.with_padding(None);
        let query = "comment publier un post Instagram";
        let added_tokens = [
            "tabs\t\tand\t\t\tmore, a literal <eos> and <unused3>, then a <mask> and  <mask>",
            "<b>gras</b> et <code>let x = 1;</code>\n\n\nfin",
            " espace en tête et en fin ",
        ];
        for sample in SAMPLES.iter().chain(added_tokens.iter()) {
            let expected = reference
                .encode((query, *sample), true)
                .unwrap()
                .get_ids()
                .to_vec();
            assert_eq!(ours.encode_pair(query, sample, 512), expected, "sample {:?}", sample);
        }
        let long = "une longue mémoire sur la mise en production ".repeat(400);
        let expected = reference.encode((query, long.as_str()), true).unwrap().get_ids().to_vec();
        let ids = ours.encode_pair(query, &long, 512);
        assert_eq!(ids.len(), 512);
        assert_eq!(ids, expected);
    }
}
