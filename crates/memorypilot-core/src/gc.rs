/// MemoryPilot v3.0 — Garbage Collection & Memory Compression.
/// Heuristic-based cleanup: merges old low-importance memories, keeps base dense.
/// Runs as background thread or on-demand via tool.

/// Result of a GC cycle.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GcReport {
    pub expired_removed: usize,
    pub groups_merged: usize,
    pub memories_compressed: usize,
    pub orphan_links_removed: usize,
    pub db_size_before: u64,
    pub db_size_after: u64,
    pub preview_mode: bool,
    pub preview_candidates: Vec<GcPreviewCandidate>,
    pub hygiene: HygieneReport,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GcPreviewCandidate {
    pub kind: String,
    pub project: Option<String>,
    pub memory_ids: Vec<String>,
    pub sample_contents: Vec<String>,
    pub confidence_score: f64,
    pub gc_score_avg: f64,
    pub age_days_min: i64,
    pub age_days_max: i64,
    pub importance_min: i32,
    pub importance_max: i32,
}

#[derive(Debug, Clone, serde::Serialize, Default)]
pub struct HygieneReport {
    pub projects_missing_path: usize,
    pub memory_project_mismatches: usize,
    pub never_accessed_memories: usize,
    pub stale_low_value_memories: usize,
    pub orphan_entities: usize,
    pub orphan_links: usize,
    pub credential_memories: usize,
    pub global_memories: usize,
}

/// Configuration for GC behavior.
pub struct GcConfig {
    /// Memories older than this (days) with importance < threshold are candidates.
    pub age_days: i64,
    /// Importance threshold: memories below this are candidates for merge.
    pub importance_threshold: i32,
    /// Maximum memories in a merge group.
    pub max_merge_group: usize,
    /// Kinds eligible for compression.
    pub compressible_kinds: Vec<String>,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            age_days: 30,
            importance_threshold: 3,
            max_merge_group: 10,
            compressible_kinds: vec!["bug".into(), "snippet".into(), "note".into(), "todo".into()],
        }
    }
}

/// Merge a group of related old memories into a single condensed memory.
/// Pure heuristic summarization — no LLM needed.
pub fn merge_memories(contents: &[String], kind: &str, project: Option<&str>) -> String {
    if contents.len() == 1 {
        return contents[0].clone();
    }

    // Count word frequency across all memories (document frequency, not raw)
    let mut word_freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for c in contents {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for w in c.split_whitespace() {
            let w = w
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if w.len() > 3 && !is_stopword(&w) && seen.insert(w.clone()) {
                *word_freq.entry(w).or_default() += 1;
            }
        }
    }

    // Top 5 keywords = subject
    let mut top_words: Vec<(String, usize)> = word_freq.into_iter().collect();
    top_words.sort_by(|a, b| b.1.cmp(&a.1));
    let subject: String = top_words
        .iter()
        .take(5)
        .map(|(w, _)| w.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    // Build condensed summary
    let project_prefix = project.map(|p| format!("[{}] ", p)).unwrap_or_default();
    let kind_label = match kind {
        "bug" => "Bugs",
        "snippet" => "Code snippets",
        "note" => "Notes",
        "todo" => "TODOs",
        _ => "Items",
    };

    // Take first sentence of each memory as bullet point
    let bullets: Vec<String> = contents
        .iter()
        .filter_map(|c| {
            let trimmed = c.trim();
            // Take first sentence or first 120 chars
            let sentence = match trimmed.find(". ") {
                Some(i) => &trimmed[..i + 1],
                None => crate::text::prefix(trimmed, 120),
            };
            if sentence.len() > 5 {
                Some(format!("- {}", sentence))
            } else {
                None
            }
        })
        .take(8) // Max 8 bullets
        .collect();

    format!(
        "{}[MERGED] {} related to: {}. ({} items compressed)\n{}",
        project_prefix,
        kind_label,
        subject,
        contents.len(),
        bullets.join("\n")
    )
}

/// Score a memory for GC candidacy (higher = more likely to be collected).
/// Returns 0.0-1.0.
pub fn gc_score(importance: i32, age_days: i64, kind: &str, _config: &GcConfig) -> f64 {
    // Base score from importance (lower importance = higher GC score)
    let importance_score = 1.0 - ((importance as f64 - 1.0) / 4.0); // 1->1.0, 5->0.0

    // Age factor (older = higher score)
    let age_factor = (age_days as f64 / 365.0).min(1.0);

    // Kind weight (some kinds are more expendable)
    let kind_weight = match kind {
        "todo" => 1.2,       // Completed/stale todos are prime candidates
        "bug" => 1.0,        // Old bugs are likely resolved
        "note" => 0.9,       // Notes may be transient
        "snippet" => 0.6,    // Snippets are often reusable
        "decision" => 0.3,   // Decisions are important context
        "preference" => 0.2, // Preferences should persist
        "pattern" => 0.2,    // Patterns are reusable
        "skill" => 0.1,      // Skills are curated procedures
        "fact" => 0.4,       // Facts may become outdated
        "credential" => 0.1, // Credentials should persist
        _ => 0.5,
    };

    (importance_score * 0.4 + age_factor * 0.3 + kind_weight * 0.3).min(1.0)
}

// ─── Auto-compaction threshold ────────────────────────

/// Threshold at which auto-compaction triggers (number of memories).
pub const AUTO_COMPACT_THRESHOLD: usize = 500;

/// Generate a memory capsule from a set of old memories.
/// Ultra-condensed summary (100-200 tokens) preserving key facts and KG links.
pub fn capsule_summary(contents: &[String], kinds: &[String], project: Option<&str>) -> String {
    if contents.len() == 1 {
        return contents[0].clone();
    }

    let mut word_freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut fact_sentences: Vec<String> = Vec::new();

    for c in contents {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for w in c.split_whitespace() {
            let w = w
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();
            if w.len() > 3 && !is_stopword(&w) && seen.insert(w.clone()) {
                *word_freq.entry(w).or_default() += 1;
            }
        }
        let trimmed = c.trim();
        let sentence = match trimmed.find(". ") {
            Some(i) => &trimmed[..i + 1],
            None => crate::text::prefix(trimmed, 100),
        };
        if sentence.len() > 5 {
            fact_sentences.push(sentence.to_string());
        }
    }

    let mut top_words: Vec<(String, usize)> = word_freq.into_iter().collect();
    top_words.sort_by(|a, b| b.1.cmp(&a.1));
    let keywords: String = top_words
        .iter()
        .take(8)
        .map(|(w, _)| w.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    let unique_kinds: std::collections::HashSet<&str> = kinds.iter().map(|k| k.as_str()).collect();
    let kinds_label: String = unique_kinds.into_iter().collect::<Vec<_>>().join("+");

    let project_prefix = project.map(|p| format!("[{}] ", p)).unwrap_or_default();

    let bullets: Vec<String> = fact_sentences
        .iter()
        .take(6)
        .map(|s| format!("- {}", s))
        .collect();

    format!(
        "{}[CAPSULE:{}] {} items | keywords: {}\n{}",
        project_prefix,
        kinds_label,
        contents.len(),
        keywords,
        bullets.join("\n")
    )
}

/// Estimate importance of content based on heuristic patterns.
/// Returns (importance 1-5, inferred_kind, suggested_ttl_days).
pub fn auto_classify(content: &str) -> (i32, &'static str, Option<i64>) {
    let lower = content.to_lowercase();
    let any = |terms: &[&str]| terms.iter().any(|term| has_term(&lower, term));

    // Only an actual secret value makes a credential: a credential is hidden from
    // safe recall, so "secrets go to 1Password" or "90k tokens" must stay visible.
    if crate::db::contains_secret_value(content) {
        return (5, "credential", None);
    }

    // Conversation chunks are useful for retrieval, but should not crowd durable facts.
    let corpus = crate::graph::analyze_corpus(content, None);
    if corpus.origin == "ai_transcript"
        || lower.trim_start().starts_with("user:")
        || lower.trim_start().starts_with("assistant:")
    {
        return (2, "transcript", Some(180));
    }

    let mentions_stack = has_term(&lower, "stack") && !lower.contains("stack trace");
    if mentions_stack
        || any(&["architecture", "we decided", "on a décidé", "migration", "migrations", "breaking change"])
    {
        return (5, "architecture", None);
    }

    // Resource identifiers are what later sessions look up most.
    if any(&[
        "project id", "project_id", "project ref", "app id", "bundle id", "team id", "account id",
        "price id", "bucket", "namespace", "database id", "client id", "org id", "organization id",
    ]) {
        return (5, "fact", None);
    }

    if any(&[
        "decided", "decision", "décision", "décidé", "décidée", "convention", "approach", "design pattern",
        "choix", "tranché",
    ]) {
        return (4, "decision", None);
    }

    if any(&[
        "prefer", "prefers", "always use", "never use", "préfère", "préférence", "toujours utiliser",
        "jamais utiliser", "i like", "j'aime",
    ]) {
        return (4, "preference", None);
    }

    if any(&[
        "pattern", "patterns", "best practice", "rule", "rules", "règle", "règles", "standard", "guideline",
        "guidelines",
    ]) {
        return (4, "pattern", None);
    }

    // Bug (importance 3, TTL 90 days — bugs get fixed)
    if any(&[
        "bug", "bugs", "error", "errors", "erreur", "erreurs", "fix", "fixed", "fixes", "corrigé", "corrigée",
        "correctif", "crash", "crashes", "plantage", "broken", "exception", "stack trace", "régression",
    ]) {
        return (3, "bug", Some(90));
    }

    // Todo (importance 2, TTL 30 days)
    if any(&["todo", "à faire", "task", "tasks", "implement", "need to", "il faut", "should add", "reste à"]) {
        return (2, "todo", Some(30));
    }

    if looks_like_code(content) {
        return (2, "snippet", None);
    }

    if any(&[
        "shipped", "deployed", "launched", "released", "milestone", "livré", "livrée", "déployé", "déployée",
        "mis en production", "mise en production", "v1", "v2",
    ]) {
        return (4, "milestone", None);
    }

    // Default: fact, importance 3
    (3, "fact", None)
}

/// `term` appears as whole words ("fix" in "a fix", not in "prefix").
fn has_term(text: &str, term: &str) -> bool {
    let is_boundary = |c: Option<char>| !matches!(c, Some(c) if c.is_alphanumeric() || c == '_');
    let mut start = 0;
    while let Some(pos) = text[start..].find(term) {
        let at = start + pos;
        let end = at + term.len();
        if is_boundary(text[..at].chars().next_back()) && is_boundary(text[end..].chars().next()) {
            return true;
        }
        start = end;
    }
    false
}

/// Code fence, or a line that starts like a declaration and carries code punctuation.
/// Prose such as "export FEC du mois" or "import bancaire" is not code.
fn looks_like_code(content: &str) -> bool {
    if content.contains("```") {
        return true;
    }
    const STARTS: &[&str] = &[
        "fn ", "pub fn ", "async fn ", "function ", "class ", "import ", "export ", "const ", "let ", "def ",
        "struct ", "impl ", "interface ", "type ",
    ];
    content.lines().map(str::trim_start).any(|line| {
        STARTS.iter().any(|s| line.starts_with(s)) && line.contains(['(', '{', '=', ';'])
    })
}

#[cfg(test)]
mod tests {
    use super::auto_classify;

    #[test]
    fn mentions_of_secrets_are_not_credentials() {
        for content in [
            "Docix : secrets Cloudflare pour la prod, référence 1Password via op-sof, jamais de secret dans git.",
            "Le catalogue MCP coûte ~90k tokens par tour.",
            "Clé Anthropic du banc : op://Soflution/yquxzdeotwopqmgf5iyzpoqgry/credential",
            "Base sauvegardée dans ~/.MemoryPilot/memory.db.bak-20261025-1444-prebuild avant le build.",
            "Binaire copié depuis /Users/me/Cursor/App/MemoryPilot/target/release/MemoryPilot2 (~ 40 Mo).",
        ] {
            assert_ne!(auto_classify(content).1, "credential", "{}", content);
        }
        assert_eq!(auto_classify("Compte test : demo@example.com / password: Hx9!kLm2pQ").1, "credential");
    }

    #[test]
    fn keywords_match_whole_words_only() {
        assert_ne!(auto_classify("Le préfixe re_ et le suffixe sont normalisés avec un prefix commun.").1, "bug");
        assert_ne!(auto_classify("Sortie de la v10.2 du moteur OCR").1, "milestone");
        assert_eq!(auto_classify("Crash au lancement : stack trace dans Sentry.").1, "bug");
        assert_eq!(auto_classify("Stack Docix : SvelteKit 2, Supabase, Cloudflare Pages.").1, "architecture");
    }

    #[test]
    fn prose_about_exports_is_not_a_snippet() {
        assert_ne!(auto_classify("Export FEC du mois envoyé au comptable, import bancaire Powens OK.").1, "snippet");
        assert_eq!(auto_classify("const total = lines.reduce((a, b) => a + b, 0);").1, "snippet");
    }

    #[test]
    fn identifiers_are_important_facts() {
        assert_eq!(auto_classify("Sociomator iOS : App ID 6758438203, Bundle ID com.soflution.pubflow").0, 5);
    }
}

/// Common English/French stopwords to skip during keyword extraction.
fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        // English
        "the" | "this" | "that" | "with" | "from" | "have" | "been" | "will"
        | "should" | "would" | "could" | "when" | "where" | "what" | "which"
        | "their" | "there" | "they" | "them" | "then" | "than" | "these"
        | "those" | "into" | "some" | "such" | "also" | "does"
        | "done" | "each" | "just" | "like" | "make" | "made" | "more"
        | "most" | "much" | "need" | "only" | "over" | "very" | "well"
        | "about" | "after" | "again" | "being" | "other" | "using"
        // French
        | "dans" | "pour" | "avec" | "cette" | "sont" | "mais" | "plus"
        | "tout" | "tous" | "toute" | "comme" | "faire" | "fait" | "peut"
        | "sans" | "encore" | "entre" | "aussi" | "autre" | "avant"
    )
}
