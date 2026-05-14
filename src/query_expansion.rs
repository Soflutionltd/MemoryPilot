//! Lexical query expansion for the BM25 lane.
//!
//! Stemming alone cannot bridge synonym pairs whose roots differ
//! (e.g. French "pondérer" / "poids", "fuite" / "leak", or English
//! "warmup" / "boot"). This module adds a small, curated bilingual
//! technical thesaurus that the FTS5 layer uses to widen the recall
//! net without touching the embedding pipeline.
//!
//! Design choices:
//! - The thesaurus is intentionally small and tech-flavoured. We add
//!   pairs that we have measurably seen biting on real queries
//!   (driven by the FR bench failures). It is deliberately not a
//!   general-purpose synonym engine: a wide thesaurus would degrade
//!   BM25 precision on unrelated queries.
//! - Lookups are bidirectional: a hit in either direction expands.
//! - Output tokens are lowercased and deduplicated. The caller is
//!   expected to feed them as additional FTS5 OR terms.
//!
//! Status: **dormant**. Three integration strategies were measured on
//! the FR bench (see `db.rs::search`); all three regressed precision
//! on small corpora because the curated thesaurus is too generic.
//! The cross-encoder reranker (jina-v2-multilingual, mode `adaptive`)
//! turned out to be the right tool for cross-lingual recall and
//! ships enabled by default. We keep this module compiled so callers
//! that want to experiment (long-tail corpora, debugging) can still
//! invoke it without re-introducing dead code, and so the test suite
//! continues to enforce the dictionary's contract.
//!
//! Roadmap: a future iteration can drive expansions from the local
//! knowledge graph (`memory_entities` co-occurrences) and from
//! pseudo-relevance feedback on the top-3 BM25 hits, but until that
//! exists this module stays silent on the search hot path.

#![allow(dead_code)]

use std::collections::HashSet;
use std::sync::OnceLock;

/// Bilingual tech thesaurus. Each row is a cluster of words/phrases
/// that should be treated as synonyms for the purpose of BM25 recall.
/// Lowercase, ASCII-stemmable lemmas only (we never insert raw
/// inflections; the stemmer takes care of those).
const THESAURUS: &[&[&str]] = &[
    // FR/EN technical synonyms surfaced by the memorypilot-fr-30 bench.
    &["pondérer", "pondération", "poids", "weight", "weighted"],
    &["score", "scores", "baseline", "métrique", "metric", "metrics"],
    &["fuite", "leak", "exposition", "exposed"],
    &["démarrage", "boot", "startup", "lancement", "launch"],
    &["warm-up", "warmup", "préchauffage", "warm"],
    &["arrêt", "shutdown", "stop", "halt"],
    &["diversifier", "diversité", "diversification", "diversity", "diverse"],
    &["fusion", "merge", "fusionner", "combiner", "combine"],
    &["doublon", "doublons", "duplicate", "duplicates", "dedup", "déduplication", "dédup"],
    &["nettoyage", "cleanup", "purge", "garbage collection", "gc"],
    &["expirer", "expiré", "expired", "ttl", "expires"],
    &["sécurité", "security", "secure", "safe"],
    &["confiance", "trust", "trusted"],
    &["mode safe", "safe mode", "mode sûr"],
    &["credentials", "credential", "secret", "secrets", "clé api", "api key", "token"],
    &["scope", "scopé", "scoped", "portée", "isolation", "isolé"],
    &["session", "thread", "fenêtre", "window", "panel"],
    &["scratchpad", "brouillon", "éphémère", "ephemeral"],
    &["référence", "ref", "baseline", "reference"],
    &["concurrence", "concurrency", "parallélisme", "parallelism", "parallèle", "parallel"],
    &["bottleneck", "goulot", "engorgement"],
    &["throughput", "débit"],
    &["latence", "latency"],
    &["embedding", "embeddings", "vecteur", "vector", "représentation", "representation"],
    &["quantisation", "quantization", "quantifier", "compress", "compression"],
    &["recherche", "search", "query", "requête"],
    &["retrieval", "rappel", "recall"],
    &["index", "indexation", "indexing"],
    &["fichier", "file", "path", "chemin"],
    &["base de données", "database", "db", "stockage", "storage"],
    &["migration", "migrer", "upgrade", "mise à jour", "update"],
    &["tests", "test", "validation", "valider", "validate"],
    &["bug", "défaut", "erreur", "error", "issue"],
    &["correction", "fix", "patch", "résolution", "resolve"],
    &["déploiement", "deploy", "deployment", "déployer", "production", "prod"],
    &["log", "logs", "journalisation", "telemetry", "télémétrie"],
    &["observabilité", "observability", "monitoring", "monitor"],
    &["benchmark", "bench", "mesure", "measurement"],
    &["régression", "regression"],
    &["pipeline", "chaîne", "workflow"],
    &["documentation", "docs", "doc"],
    &["api", "endpoint", "interface"],
    &["webhook", "callback", "rappel http"],
    &["signature", "sig", "hmac"],
    &["authentication", "auth", "authentification", "login", "connexion"],
    &["autorisation", "authorization", "permission"],
    &["row level security", "rls", "policy", "politique"],
    &["tenant", "client", "locataire", "multi-tenant"],
    &["frontend", "ui", "interface utilisateur", "front"],
    &["backend", "serveur", "server", "back"],
    &["framework", "librairie", "library", "lib"],
    &["composant", "component", "widget"],
    &["état", "state", "store", "stockage local"],
    &["dérivation", "derived", "computed", "calculé"],
    &["effet", "effect", "side effect", "effet de bord"],
    &["réactif", "reactive", "reactivity", "réactivité"],
    &["compilation", "build", "compile", "compiler", "bundling"],
    &["déboguer", "debug", "déboggage", "debugging"],
    &["profilage", "profiling", "profiler"],
    &["concurrent", "simultané", "parallèle"],
    &["bloquant", "blocking", "synchrone", "sync"],
    &["non bloquant", "non-blocking", "async", "asynchrone", "asynchronous"],
];

fn thesaurus_index() -> &'static std::collections::HashMap<String, Vec<&'static str>> {
    static INDEX: OnceLock<std::collections::HashMap<String, Vec<&'static str>>> =
        OnceLock::new();
    INDEX.get_or_init(|| {
        let mut map: std::collections::HashMap<String, Vec<&'static str>> =
            std::collections::HashMap::new();
        for cluster in THESAURUS {
            for term in cluster.iter() {
                let normalized = term.to_lowercase();
                map.entry(normalized)
                    .or_default()
                    .extend(cluster.iter().copied());
            }
        }
        map
    })
}

/// Expand a query into the set of additional terms that should be ORed
/// into the FTS5 BM25 layer. The original query tokens are NOT
/// included; the caller already searches for them. Returns a
/// deduplicated, lowercased set, capped at 12 terms to keep the FTS5
/// query budget reasonable.
pub fn expand(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    let index = thesaurus_index();
    let mut additions: HashSet<String> = HashSet::new();

    // Direct token lookups.
    for token in lower.split(|c: char| !c.is_alphanumeric()) {
        if token.len() < 3 {
            continue;
        }
        if let Some(cluster) = index.get(token) {
            for term in cluster {
                let normalized = term.to_lowercase();
                if normalized != token {
                    additions.insert(normalized);
                }
            }
        }
    }

    // Multi-word phrases: scan for known multi-word keys.
    for key in index.keys() {
        if key.contains(' ') && lower.contains(key.as_str()) {
            if let Some(cluster) = index.get(key) {
                for term in cluster {
                    let normalized = term.to_lowercase();
                    if normalized != *key {
                        additions.insert(normalized);
                    }
                }
            }
        }
    }

    let mut result: Vec<String> = additions.into_iter().collect();
    result.sort();
    result.truncate(12);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_pondering_to_weight_synonyms() {
        let terms = expand("comment pondérer les colonnes dans BM25");
        assert!(terms.iter().any(|t| t == "poids"), "expected 'poids' in {:?}", terms);
        assert!(terms.iter().any(|t| t == "weight"), "expected 'weight' in {:?}", terms);
    }

    #[test]
    fn expands_baseline_query() {
        let terms = expand("scores de référence sur le benchmark LongMemEval");
        assert!(terms.iter().any(|t| t == "baseline"), "expected 'baseline' in {:?}", terms);
    }

    #[test]
    fn expands_session_diversity() {
        let terms = expand("diversifier les résultats par session pour éviter la concentration");
        assert!(terms.iter().any(|t| t == "diversity" || t == "diversité"));
    }

    #[test]
    fn empty_query_returns_no_expansions() {
        assert!(expand("").is_empty());
    }

    #[test]
    fn unrelated_query_yields_no_noise() {
        let terms = expand("aaaa bbbb cccc");
        assert!(terms.is_empty(), "expected no expansion, got {:?}", terms);
    }

    #[test]
    fn expansion_is_capped() {
        let terms = expand("session embedding query database test fix bench api auth deploy log security");
        assert!(terms.len() <= 12, "expansion not capped, got {} terms", terms.len());
    }
}
