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
            let end = trimmed
                .find(". ")
                .map(|i| i + 1)
                .unwrap_or_else(|| trimmed.len().min(120));
            let sentence = &trimmed[..end];
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
        let end = trimmed
            .find(". ")
            .map(|i| i + 1)
            .unwrap_or_else(|| trimmed.len().min(100));
        if end > 5 {
            fact_sentences.push(trimmed[..end].to_string());
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
    let corpus = crate::graph::analyze_corpus(content, None);

    // Credential/secret detection (importance 5, no TTL)
    if lower.contains("api_key")
        || lower.contains("api key")
        || lower.contains("secret")
        || lower.contains("password")
        || lower.contains("token")
        || lower.contains("credential")
        || lower.contains("private_key")
    {
        return (5, "credential", None);
    }

    // Architecture/decision (importance 4-5)
    if lower.contains("architecture")
        || lower.contains("we decided")
        || lower.contains("on a décidé")
        || lower.contains("stack")
        || lower.contains("migration")
        || lower.contains("breaking change")
    {
        return (5, "architecture", None);
    }
    if lower.contains("decided")
        || lower.contains("decision")
        || lower.contains("décision")
        || lower.contains("convention")
        || lower.contains("approach")
        || lower.contains("design pattern")
    {
        return (4, "decision", None);
    }

    // Preference (importance 4)
    if lower.contains("prefer")
        || lower.contains("always use")
        || lower.contains("never use")
        || lower.contains("préfère")
        || lower.contains("toujours utiliser")
        || lower.contains("jamais utiliser")
        || lower.contains("i like")
        || lower.contains("j'aime")
    {
        return (4, "preference", None);
    }

    // Pattern (importance 3-4)
    if lower.contains("pattern")
        || lower.contains("best practice")
        || lower.contains("convention")
        || lower.contains("rule")
        || lower.contains("standard")
        || lower.contains("guideline")
    {
        return (4, "pattern", None);
    }

    // Bug (importance 3, TTL 90 days — bugs get fixed)
    if lower.contains("bug")
        || lower.contains("error")
        || lower.contains("fix")
        || lower.contains("crash")
        || lower.contains("broken")
        || lower.contains("erreur")
        || lower.contains("exception")
        || lower.contains("stack trace")
    {
        return (3, "bug", Some(90));
    }

    // Todo (importance 2, TTL 30 days)
    if lower.contains("todo")
        || lower.contains("à faire")
        || lower.contains("task")
        || lower.contains("implement")
        || lower.contains("need to")
        || lower.contains("il faut")
        || lower.contains("should add")
    {
        return (2, "todo", Some(30));
    }

    // Snippet (importance 2)
    if lower.contains("```")
        || lower.contains("fn ")
        || lower.contains("function ")
        || lower.contains("class ")
        || lower.contains("import ")
        || lower.contains("const ")
        || lower.contains("export ")
    {
        return (2, "snippet", None);
    }

    // Milestone (importance 4)
    if lower.contains("shipped")
        || lower.contains("deployed")
        || lower.contains("launched")
        || lower.contains("released")
        || lower.contains("milestone")
        || lower.contains("livré")
        || lower.contains("déployé")
        || lower.contains("v1")
        || lower.contains("v2")
    {
        return (4, "milestone", None);
    }

    // Conversation chunks are useful for retrieval, but should not crowd durable facts.
    if corpus.origin == "ai_transcript"
        || lower.trim_start().starts_with("user:")
        || lower.trim_start().starts_with("assistant:")
    {
        return (2, "transcript", Some(180));
    }

    // Default: fact, importance 3
    (3, "fact", None)
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
