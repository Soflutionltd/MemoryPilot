//! Procedural memory: reusable "skills" mined from recurring procedures.
//!
//! Agents keep re-describing the same procedure ("déploiement Docix :
//! cd apps/web && npm run build puis wrangler pages deploy ..."), spread
//! over several memories written on different days. A skill folds such a
//! cluster into one actionable card: when to use it, the ordered steps,
//! the pitfalls, and where it came from.
//!
//! Fully local and deterministic, no LLM:
//! 1. candidates = memories with procedural signals (commands, step
//!    markers, procedure verbs), never transcripts / credentials;
//! 2. cluster per project by cosine (stored embeddings, union-find);
//! 3. keep clusters that recur (>= 3 memories, or >= 2 on distinct days);
//! 4. extract steps / pitfalls extractively from the members' sentences;
//! 5. upsert as `kind = "skill"` memories, matched to previous runs by
//!    source overlap so ids stay stable and re-runs are near no-ops.

use std::collections::{HashMap, HashSet};

use rusqlite::params;
use serde_json::{json, Value};

use super::{Database, MemoryScope};

const SKILL_KIND: &str = "skill";
/// Looser than consolidation (0.95): two descriptions of one procedure
/// share the commands but not the wording.
const CLUSTER_THRESHOLD: f32 = 0.80;
const MIN_PROCEDURAL_SCORE: i32 = 2;
const MAX_STEPS: usize = 8;
const MAX_PITFALLS: usize = 3;
const MAX_SKILLS: usize = 80;
const AUTO_INTERVAL_SECS: i64 = 24 * 3600;
const AUTO_LAST_RUN_KEY: &str = "last_skill_extract_at";

const SOURCE_KINDS: &[&str] = &[
    "decision",
    "fact",
    "pattern",
    "milestone",
    "bug",
    "architecture",
    "note",
    "snippet",
    "problem",
];

const COMMAND_MARKERS: &[&str] = &[
    "cargo build",
    "cargo run",
    "cargo test",
    "cargo install",
    "npm run",
    "npm install",
    "npm i ",
    "npm ci",
    "npx ",
    "pnpm ",
    "yarn ",
    "git push",
    "git pull",
    "git commit",
    "git clone",
    "git checkout",
    "git merge",
    "git tag",
    "codesign",
    "xcodebuild",
    "xcrun",
    "notarytool",
    "wrangler",
    "supabase db",
    "supabase functions",
    "supabase link",
    "brew install",
    "sqlite3 ",
    "osascript",
    "launchctl",
    "docker build",
    "docker run",
    "docker compose",
    "make install",
    "make release",
    "make build",
    "flutter build",
    "flutter run",
    "flutter pub",
    "flutter clean",
    "pod install",
    "curl -",
    "ssh ",
    "rsync",
    "chmod ",
    "defaults write",
    "tauri build",
    "tauri dev",
    "python3 ",
    "node ",
    "deno ",
    "bun run",
    "go build",
    "swift build",
    "devicectl",
    "gh release",
    "gh pr",
    "fastlane",
    "cursor-helper",
    "hdiutil",
];
const STEP_MARKERS: &[&str] = &[
    " puis ",
    " ensuite ",
    " then ",
    "->",
    "→",
    "étape",
    "step ",
    "1.",
    "1)",
    "&&",
];
const PROCEDURE_VERBS: &[&str] = &[
    "déploi",
    "deploy",
    "install",
    "build",
    "compil",
    "release",
    "publi",
    "configur",
    "relanc",
    "lancer",
    "signer",
    "sign ",
    "migrat",
    "backup",
    "sauvegard",
    "restaur",
    "mettre à jour",
    "update",
    "upgrade",
    "rebuild",
    "redémarr",
    "restart",
];
const PITFALL_MARKERS: &[&str] = &[
    "ne pas",
    "n'utiliser",
    "jamais",
    "piège",
    "attention",
    "faux",
    "never",
    "don't",
    "do not",
    "warning",
    "⚠",
    "surtout pas",
    "éviter",
    "avoid",
    "erreur",
    "error",
    "échoue",
    "fails",
];
const STOPWORDS: &[&str] = &[
    "pour",
    "dans",
    "avec",
    "sans",
    "mais",
    "plus",
    "tout",
    "tous",
    "cette",
    "sont",
    "être",
    "fait",
    "faire",
    "puis",
    "ensuite",
    "aussi",
    "comme",
    "depuis",
    "entre",
    "leur",
    "leurs",
    "donc",
    "alors",
    "quand",
    "that",
    "this",
    "with",
    "from",
    "have",
    "then",
    "when",
    "into",
    "were",
    "will",
    "your",
    "also",
    "only",
    "used",
    "uses",
    "using",
    "après",
    "avant",
    "chez",
    "très",
    "elle",
    "nous",
    "vous",
    "sous",
    "dont",
    "était",
    "sera",
    "utilisateur",
    "user",
];

struct Candidate {
    id: String,
    project: String,
    content: String,
    created_at: String,
    vector: Vec<f32>,
}

struct SkillDraft {
    project: String,
    title: String,
    content: String,
    source_ids: Vec<String>,
    occurrences: usize,
    first_seen: String,
    last_seen: String,
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

/// How much a text looks like a procedure (commands, steps, verbs).
pub(crate) fn procedural_score(content: &str) -> i32 {
    let lower = content.to_lowercase();
    let mut score = 0;
    if contains_any(&lower, COMMAND_MARKERS) {
        score += 2;
    }
    if contains_any(&lower, STEP_MARKERS) {
        score += 1;
    }
    if contains_any(&lower, PROCEDURE_VERBS) {
        score += 1;
    }
    score
}

fn has_command(text: &str) -> bool {
    text.contains('`') || contains_any(&text.to_lowercase(), COMMAND_MARKERS)
}

/// Infinitive / imperative openings that make a sentence an instruction
/// rather than a status report ("Slate 0.3.138 publiée").
const INSTRUCTION_STARTS: &[&str] = &[
    "lancer",
    "relancer",
    "signer",
    "compiler",
    "déployer",
    "deployer",
    "publier",
    "installer",
    "exécuter",
    "executer",
    "ouvrir",
    "copier",
    "configurer",
    "ajouter",
    "supprimer",
    "vérifier",
    "verifier",
    "mettre",
    "utiliser",
    "pousser",
    "créer",
    "creer",
    "commande",
    "run ",
    "build ",
    "deploy ",
    "install ",
    "sign ",
    "open ",
    "copy ",
    "set ",
    "use ",
    "push ",
    "create ",
    "check ",
    "cd ",
    "ne pas",
    "toujours",
    "always",
];

const STATUS_MARKERS: &[&str] = &[
    "publiée",
    "publié ",
    "published",
    "baseline",
    "réussi",
    "succeeded",
    "déployé sur",
    "terminé",
    "a été",
    "was archived",
    "validations ok",
];

fn root(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn unite(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (root(parent, a), root(parent, b));
    if ra != rb {
        parent[rb] = ra;
    }
}

/// Split a memory into sentence-like segments, keeping commands intact.
fn segments(content: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in content.lines() {
        let mut rest = line.trim();
        // Bracketed prefixes like "[CompressKit - Fix ...]" are titles, not steps.
        if rest.starts_with('[') {
            if let Some(end) = rest.find(']') {
                rest = rest[end + 1..].trim();
            }
        }
        let mut current = String::new();
        let mut in_code = false;
        let chars: Vec<char> = rest.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c == '`' {
                in_code = !in_code;
            }
            current.push(c);
            let boundary = !in_code
                && (c == ';'
                    || (c == '.'
                        && chars.get(i + 1).map(|n| n.is_whitespace()).unwrap_or(true)
                        && !current
                            .trim_end_matches('.')
                            .ends_with(|ch: char| ch.is_ascii_digit())));
            if boundary {
                let piece = current
                    .trim()
                    .trim_end_matches([';', '.'])
                    .trim()
                    .to_string();
                if !piece.is_empty() {
                    out.push(piece);
                }
                current.clear();
            }
            i += 1;
        }
        let piece = current
            .trim()
            .trim_end_matches([';', '.'])
            .trim()
            .to_string();
        if !piece.is_empty() {
            out.push(piece);
        }
    }
    // "A puis B" -> two steps
    out.into_iter()
        .flat_map(|segment| {
            segment
                .split(" puis ")
                .flat_map(|part| part.split(", then "))
                .map(|part| part.trim().to_string())
                .filter(|part| part.chars().count() >= 8)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn clip(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut clipped: String = text.chars().take(max_chars).collect();
    if let Some(space) = clipped.rfind(' ') {
        if space > max_chars / 2 {
            clipped.truncate(space);
        }
    }
    format!("{}…", clipped.trim_end())
}

fn title_of(content: &str) -> String {
    let mut text = content.trim();
    if text.starts_with('[') {
        if let Some(end) = text.find(']') {
            let inner = text[1..end].trim();
            if inner.chars().count() >= 6 {
                return clip(inner, 80);
            }
            text = text[end + 1..].trim();
        }
    }
    let cut = text
        .find(|c: char| c == ':' || c == '\n' || c == '(')
        .unwrap_or(text.len());
    let head = text[..cut].trim();
    let head = if head.chars().count() >= 6 {
        head
    } else {
        text
    };
    let head = head.split(". ").next().unwrap_or(head);
    clip(head, 80)
}

/// Remove what changes between two runs of the same procedure: dates,
/// version numbers, build numbers ("Déploiement du 2026-03-06",
/// "Slate 0.3.135 publiée").
fn strip_volatile(title: &str) -> String {
    let mut words: Vec<&str> = Vec::new();
    let tokens: Vec<&str> = title.split_whitespace().collect();
    let mut i = 0;
    while i < tokens.len() {
        let token = tokens[i];
        let core = token.trim_matches(|c: char| !c.is_alphanumeric());
        let digits = core.chars().filter(|c| c.is_ascii_digit()).count();
        let volatile = !core.is_empty()
            && (digits * 2 >= core.chars().count()
                || (core.starts_with('v')
                    && core[1..]
                        .chars()
                        .next()
                        .map(|c| c.is_ascii_digit())
                        .unwrap_or(false)));
        let lower = core.to_lowercase();
        let connector = matches!(
            lower.as_str(),
            "du" | "le" | "la" | "on" | "of" | "build" | "version" | "v"
        );
        if volatile {
            // drop a dangling connector before the volatile token ("du 2026-03-06")
            if let Some(last) = words.last() {
                let l = last.to_lowercase();
                if matches!(
                    l.as_str(),
                    "du" | "le"
                        | "la"
                        | "on"
                        | "of"
                        | "build"
                        | "version"
                        | "effectué"
                        | "effectuée"
                ) {
                    words.pop();
                }
            }
            i += 1;
            continue;
        }
        if connector
            && tokens
                .get(i + 1)
                .map(|n| n.chars().any(|c| c.is_ascii_digit()))
                .unwrap_or(false)
        {
            i += 1;
            continue;
        }
        words.push(token);
        i += 1;
    }
    words
        .join(" ")
        .trim_matches(|c: char| c == '-' || c == ':' || c.is_whitespace())
        .to_string()
}

/// Category of procedure, from the vocabulary shared by the members.
fn category_label(members: &[&Candidate]) -> Option<&'static str> {
    let text = members
        .iter()
        .map(|m| m.content.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    const CATEGORIES: &[(&[&str], &str)] = &[
        (
            &[
                "testflight",
                "app store connect",
                "resubmission",
                "xcodebuild archive",
                "altool",
            ],
            "Publier sur l'App Store / TestFlight",
        ),
        (
            &["wrangler", "pages deploy", "cloudflare pages"],
            "Déployer sur Cloudflare",
        ),
        (&["vercel"], "Déployer sur Vercel"),
        (
            &["notarytool", "notaris", "codesign"],
            "Signer et notariser l'app macOS",
        ),
        (
            &[".dmg", "release", "publiée", "publié", "published"],
            "Publier une nouvelle version",
        ),
        (&["migrat"], "Migrer"),
        (&["déploi", "deploy"], "Déployer"),
        (
            &["cargo build", "npm run build", "xcodebuild", "compil"],
            "Compiler",
        ),
    ];
    CATEGORIES
        .iter()
        .find(|(needles, _)| needles.iter().filter(|n| text.contains(*n)).count() >= 1)
        .map(|(_, label)| *label)
}

fn keywords(members: &[&Candidate], title: &str) -> Vec<String> {
    let title_lower = title.to_lowercase();
    let mut freq: HashMap<String, (usize, usize)> = HashMap::new();
    for (order, member) in members.iter().enumerate() {
        let mut seen = HashSet::new();
        for raw in member
            .content
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
        {
            let word = raw.trim_matches(|c: char| c == '-' || c == '_');
            if word.chars().count() < 4 || word.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            if STOPWORDS.contains(&word) || title_lower.contains(word) {
                continue;
            }
            if seen.insert(word.to_string()) {
                let entry = freq.entry(word.to_string()).or_insert((0, order));
                entry.0 += 1;
            }
        }
    }
    let mut words: Vec<(String, (usize, usize))> = freq
        .into_iter()
        .filter(|(_, (count, _))| *count >= 2 || members.len() <= 2)
        .collect();
    words.sort_by(|a, b| {
        b.1 .0
            .cmp(&a.1 .0)
            .then(a.1 .1.cmp(&b.1 .1))
            .then(a.0.cmp(&b.0))
    });
    words.into_iter().take(6).map(|(word, _)| word).collect()
}

impl Database {
    fn skill_candidates(&self, project: Option<&str>) -> Result<Vec<Candidate>, String> {
        let kinds = SOURCE_KINDS
            .iter()
            .map(|kind| format!("'{}'", kind))
            .collect::<Vec<_>>()
            .join(",");
        let mut sql = format!(
            "SELECT id, project, content, created_at, embedding FROM memories
             WHERE embedding IS NOT NULL AND kind IN ({}) AND length(content) >= 60
               AND (expires_at IS NULL OR expires_at > ?1)",
            kinds
        );
        let now = chrono::Utc::now().to_rfc3339();
        let canonical = project.and_then(Self::canonical_project_name);
        if canonical.is_some() {
            sql.push_str(" AND project = ?2");
        }
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|error| format!("skills prepare: {}", error))?;
        let map_row = |row: &rusqlite::Row| -> rusqlite::Result<Candidate> {
            let blob: Vec<u8> = row.get(4)?;
            Ok(Candidate {
                id: row.get(0)?,
                project: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                content: row.get(2)?,
                created_at: row.get(3)?,
                vector: crate::embedding::blob_to_vec(&blob),
            })
        };
        let rows: Vec<Candidate> = match canonical.as_deref() {
            Some(p) => stmt
                .query_map(params![now, p], map_row)
                .map_err(|error| format!("skills query: {}", error))?
                .flatten()
                .collect(),
            None => stmt
                .query_map(params![now], map_row)
                .map_err(|error| format!("skills query: {}", error))?
                .flatten()
                .collect(),
        };
        let dim = crate::embedding::vector_dim();
        Ok(rows
            .into_iter()
            .filter(|c| {
                c.vector.len() == dim && procedural_score(&c.content) >= MIN_PROCEDURAL_SCORE
            })
            .collect())
    }

    fn draft_skill(members: &[&Candidate]) -> Option<SkillDraft> {
        // Most central member gives the title.
        let central = members
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| {
                let score = |x: &Candidate| -> f32 {
                    members
                        .iter()
                        .map(|other| crate::embedding::cosine_similarity(&x.vector, &other.vector))
                        .sum()
                };
                score(a)
                    .partial_cmp(&score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(ib.cmp(ia))
            })
            .map(|(_, member)| *member)?;
        let raw_title = strip_volatile(&title_of(&central.content));
        let starts_lower = raw_title
            .chars()
            .next()
            .map(|c| c.is_lowercase())
            .unwrap_or(true);
        let vague = raw_title.chars().count() < 6
            || starts_lower
            || ["Après", "Avant", "Ensuite", "Puis", "After", "Then"]
                .iter()
                .any(|w| raw_title.starts_with(w));
        let title = match (category_label(members), vague) {
            (Some(label), true) => label.to_string(),
            (Some(label), false)
                if !raw_title
                    .to_lowercase()
                    .contains(&label.to_lowercase().chars().take(6).collect::<String>()) =>
            {
                format!("{} : {}", label, raw_title)
            }
            (_, false) => raw_title,
            (None, true) => return None,
        };

        // Newest description first: it holds the current procedure.
        let mut ordered: Vec<&Candidate> = members.to_vec();
        ordered.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(a.id.cmp(&b.id)));

        let mut steps: Vec<(String, String)> = Vec::new();
        let mut pitfalls: Vec<(String, String)> = Vec::new();
        for member in &ordered {
            for segment in segments(&member.content) {
                let lower = segment.to_lowercase();
                let normalized = Self::normalize(&segment);
                if contains_any(&lower, PITFALL_MARKERS) {
                    if pitfalls.len() < MAX_PITFALLS
                        && !pitfalls
                            .iter()
                            .any(|(_, n)| Self::similarity(n, &normalized) >= 0.5)
                    {
                        pitfalls.push((clip(&segment, 200), normalized));
                    }
                    continue;
                }
                let command = has_command(&segment);
                let instruction = INSTRUCTION_STARTS.iter().any(|w| lower.starts_with(w));
                let status = contains_any(&lower, STATUS_MARKERS) && !command;
                let is_step = (command || instruction) && !status;
                if is_step
                    && steps.len() < MAX_STEPS
                    && !steps
                        .iter()
                        .any(|(_, n)| Self::similarity(n, &normalized) >= 0.5)
                {
                    steps.push((clip(&segment, 220), normalized));
                }
            }
        }
        if steps.len() < 2 || !steps.iter().any(|(step, _)| has_command(step)) {
            return None;
        }

        let project = central.project.clone();
        let mut days: Vec<&str> = members
            .iter()
            .map(|m| &m.created_at[..m.created_at.len().min(10)])
            .collect();
        days.sort();
        let first_seen = days.first().map(|d| d.to_string()).unwrap_or_default();
        let last_seen = days.last().map(|d| d.to_string()).unwrap_or_default();
        let keys = keywords(members, &title);

        let mut content = String::new();
        if project.is_empty() {
            content.push_str(&format!("Compétence : {}\n", title));
        } else {
            content.push_str(&format!("Compétence ({}) : {}\n", project, title));
        }
        if !keys.is_empty() {
            content.push_str(&format!("Quand l'utiliser : {}\n", keys.join(", ")));
        }
        content.push_str("Étapes :\n");
        for (index, (step, _)) in steps.iter().enumerate() {
            content.push_str(&format!("{}. {}\n", index + 1, step));
        }
        if !pitfalls.is_empty() {
            content.push_str("Pièges :\n");
            for (pitfall, _) in &pitfalls {
                content.push_str(&format!("- {}\n", pitfall));
            }
        }
        content.push_str(&format!(
            "Vu {} fois ({} → {}).",
            members.len(),
            first_seen,
            last_seen
        ));

        let mut source_ids: Vec<String> = members.iter().map(|m| m.id.clone()).collect();
        source_ids.sort();
        Some(SkillDraft {
            project,
            title,
            content,
            source_ids,
            occurrences: members.len(),
            first_seen,
            last_seen,
        })
    }

    /// Mine recurring procedures and upsert them as `skill` memories.
    /// `apply = false` only reports what would be written.
    pub fn extract_skills(&self, apply: bool, project: Option<&str>) -> Result<Value, String> {
        let candidates = self.skill_candidates(project)?;

        let mut by_project: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, candidate) in candidates.iter().enumerate() {
            by_project
                .entry(candidate.project.as_str())
                .or_default()
                .push(index);
        }
        let mut parent: Vec<usize> = (0..candidates.len()).collect();
        for members in by_project.values() {
            for (position, &a) in members.iter().enumerate() {
                for &b in &members[position + 1..] {
                    if crate::embedding::cosine_similarity(
                        &candidates[a].vector,
                        &candidates[b].vector,
                    ) >= CLUSTER_THRESHOLD
                    {
                        unite(&mut parent, a, b);
                    }
                }
            }
        }
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for index in 0..candidates.len() {
            let r = root(&mut parent, index);
            clusters.entry(r).or_default().push(index);
        }
        let mut drafts: Vec<SkillDraft> = clusters
            .into_values()
            .filter(|members| {
                let days: HashSet<&str> = members
                    .iter()
                    .map(|&i| &candidates[i].created_at[..candidates[i].created_at.len().min(10)])
                    .collect();
                let with_commands = members
                    .iter()
                    .filter(|&&i| has_command(&candidates[i].content))
                    .count();
                with_commands >= 2 && (members.len() >= 3 || days.len() >= 2)
            })
            .filter_map(|members| {
                let refs: Vec<&Candidate> = members.iter().map(|&i| &candidates[i]).collect();
                Self::draft_skill(&refs)
            })
            .collect();
        drafts.sort_by(|a, b| {
            b.occurrences
                .cmp(&a.occurrences)
                .then(b.last_seen.cmp(&a.last_seen))
                .then(a.source_ids.cmp(&b.source_ids))
        });
        drafts.truncate(MAX_SKILLS);

        // Existing auto skills, matched by source overlap for stable ids.
        let mut existing: Vec<(String, HashSet<String>, bool)> = Vec::new();
        {
            let mut sql =
                "SELECT id, metadata, tags, project FROM memories WHERE kind = 'skill'".to_string();
            let canonical = project.and_then(Self::canonical_project_name);
            if canonical.is_some() {
                sql.push_str(" AND project = ?1");
            }
            let mut stmt = self
                .conn
                .prepare(&sql)
                .map_err(|error| format!("skills existing: {}", error))?;
            let map_row =
                |row: &rusqlite::Row| -> rusqlite::Result<(String, Option<String>, String)> {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                };
            let rows: Vec<(String, Option<String>, String)> = match canonical.as_deref() {
                Some(p) => stmt
                    .query_map(params![p], map_row)
                    .map_err(|e| e.to_string())?
                    .flatten()
                    .collect(),
                None => stmt
                    .query_map([], map_row)
                    .map_err(|e| e.to_string())?
                    .flatten()
                    .collect(),
            };
            for (id, metadata, tags) in rows {
                let meta: Value = metadata
                    .as_deref()
                    .and_then(|m| serde_json::from_str(m).ok())
                    .unwrap_or(Value::Null);
                if meta.get("auto").and_then(|v| v.as_bool()) != Some(true) {
                    continue; // hand-written skills are never touched
                }
                let sources: HashSet<String> = meta
                    .get("source_ids")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let pinned = tags.contains("pinned");
                existing.push((id, sources, pinned));
            }
        }

        let mut matched: HashSet<usize> = HashSet::new();
        let (mut added, mut updated, mut unchanged) = (0usize, 0usize, 0usize);
        let mut report_skills = Vec::new();
        for draft in &drafts {
            let draft_set: HashSet<String> = draft.source_ids.iter().cloned().collect();
            let best = existing
                .iter()
                .enumerate()
                .filter(|(i, _)| !matched.contains(i))
                .map(|(i, (_, sources, _))| {
                    let inter = sources.intersection(&draft_set).count() as f64;
                    let union = sources.union(&draft_set).count().max(1) as f64;
                    (i, inter / union)
                })
                .filter(|(_, jaccard)| *jaccard >= 0.34)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let metadata = json!({
                "auto": true,
                "title": draft.title,
                "source_ids": draft.source_ids,
                "occurrences": draft.occurrences,
                "first_seen": draft.first_seen,
                "last_seen": draft.last_seen,
            });
            let tags = vec!["skill".to_string(), "auto".to_string()];
            let mut id = None;
            let mut action = "add";
            if let Some((index, _)) = best {
                matched.insert(index);
                let existing_id = existing[index].0.clone();
                let current = self.get_memory(&existing_id)?;
                let same = current
                    .as_ref()
                    .map(|m| m.content == draft.content)
                    .unwrap_or(false);
                action = if same { "unchanged" } else { "update" };
                if apply && !same {
                    self.update_memory_full(
                        &existing_id,
                        Some(&draft.content),
                        Some(SKILL_KIND),
                        Some(&tags),
                        Some(4),
                        None,
                        Some(&metadata),
                    )?;
                }
                if same {
                    unchanged += 1;
                } else {
                    updated += 1;
                }
                id = Some(existing_id);
            } else {
                if apply {
                    let (memory, _) = self.add_memory(
                        &draft.content,
                        SKILL_KIND,
                        if draft.project.is_empty() {
                            None
                        } else {
                            Some(draft.project.as_str())
                        },
                        &tags,
                        "memorypilot-skills",
                        4,
                        None,
                        Some(&metadata),
                        &MemoryScope::default(),
                    )?;
                    id = Some(memory.id);
                }
                added += 1;
            }
            report_skills.push(json!({
                "id": id,
                "action": action,
                "project": draft.project,
                "title": draft.title,
                "occurrences": draft.occurrences,
                "last_seen": draft.last_seen,
            }));
        }

        let mut removed = 0usize;
        for (index, (id, _, pinned)) in existing.iter().enumerate() {
            if matched.contains(&index) || *pinned {
                continue;
            }
            if apply {
                let _ = self.delete_memory(id);
            }
            removed += 1;
        }

        Ok(json!({
            "applied": apply,
            "candidates_scanned": candidates.len(),
            "skills": drafts.len(),
            "added": added,
            "updated": updated,
            "unchanged": unchanged,
            "removed_stale": removed,
            "items": report_skills,
        }))
    }

    /// Compact listing of skills (auto and hand-written).
    pub fn list_skills(&self, project: Option<&str>, limit: usize) -> Result<Value, String> {
        let (memories, total) =
            self.list_memories(project, Some(SKILL_KIND), None, limit.max(1), 0)?;
        let items: Vec<Value> = memories
            .iter()
            .map(|m| {
                let meta = m.metadata.clone().unwrap_or(Value::Null);
                json!({
                    "id": m.id,
                    "project": m.project,
                    "title": meta.get("title").and_then(|v| v.as_str())
                        .map(String::from)
                        .unwrap_or_else(|| title_of(&m.content)),
                    "occurrences": meta.get("occurrences"),
                    "last_seen": meta.get("last_seen"),
                    "auto": meta.get("auto").and_then(|v| v.as_bool()).unwrap_or(false),
                })
            })
            .collect();
        Ok(json!({ "total": total, "skills": items }))
    }

    /// Full skill card plus short excerpts of the memories it came from.
    pub fn get_skill(&self, id: &str) -> Result<Option<Value>, String> {
        let Some(memory) = self.get_memory(id)? else {
            return Ok(None);
        };
        if memory.kind != SKILL_KIND {
            return Err(format!("Memory {} is a '{}', not a skill", id, memory.kind));
        }
        let meta = memory.metadata.clone().unwrap_or(Value::Null);
        let mut sources = Vec::new();
        if let Some(ids) = meta.get("source_ids").and_then(|v| v.as_array()) {
            for source_id in ids.iter().filter_map(|v| v.as_str()).take(10) {
                if let Some(source) = self.get_memory(source_id)? {
                    sources.push(json!({
                        "id": source.id,
                        "kind": source.kind,
                        "created_at": source.created_at,
                        "excerpt": clip(&source.content, 240),
                    }));
                }
            }
        }
        Ok(Some(json!({
            "id": memory.id,
            "project": memory.project,
            "content": memory.content,
            "metadata": meta,
            "sources": sources,
        })))
    }

    /// Daily automatic pass, throttled through the `config` table.
    pub(super) fn maybe_extract_skills(&self) {
        if std::env::var("MEMORYPILOT_SKILLS")
            .map(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "off" | "0" | "false"
                )
            })
            .unwrap_or(false)
        {
            return;
        }
        let now = chrono::Utc::now().timestamp();
        let last: i64 = self
            .conn
            .query_row(
                "SELECT value FROM config WHERE key=?1",
                params![AUTO_LAST_RUN_KEY],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .and_then(|raw| raw.parse().ok())
            .unwrap_or(0);
        if now - last < AUTO_INTERVAL_SECS {
            return;
        }
        let _ = self.conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?1, ?2)",
            params![AUTO_LAST_RUN_KEY, now.to_string()],
        );
        if let Ok(report) = self.extract_skills(true, None) {
            let added = report["added"].as_u64().unwrap_or(0);
            let updated = report["updated"].as_u64().unwrap_or(0);
            if added + updated > 0 {
                eprintln!(
                    "[MemoryPilot] Skills: {} new, {} updated from recurring procedures.",
                    added, updated
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_score_detects_commands_and_steps() {
        assert!(
            procedural_score(
                "Déploiement : cd apps/web && npm run build puis npx wrangler pages deploy"
            ) >= 3
        );
        assert!(procedural_score("Antoine préfère le café sans sucre") < MIN_PROCEDURAL_SCORE);
    }

    #[test]
    fn segments_split_steps_and_keep_code() {
        let parts = segments("Commande : `cd a && npm run build` puis `npx wrangler pages deploy`. NE PAS utiliser notegenius.");
        assert!(parts.iter().any(|p| p.contains("npm run build")));
        assert!(parts.iter().any(|p| p.contains("wrangler pages deploy")));
        assert!(parts.iter().any(|p| p.contains("NE PAS")));
    }

    #[test]
    fn strip_volatile_drops_dates_and_versions() {
        assert_eq!(strip_volatile("Déploiement du 2026-03-06"), "Déploiement");
        assert_eq!(strip_volatile("Slate 0.3.135 publiée"), "Slate publiée");
        assert_eq!(
            strip_volatile("PhotoPilot v1.5.17 publié"),
            "PhotoPilot publié"
        );
    }

    #[test]
    fn title_prefers_bracket_label() {
        assert_eq!(
            title_of("[CompressKit - Fix qpdf] Décision : signer"),
            "CompressKit - Fix qpdf"
        );
        assert_eq!(
            title_of("Déploiement Docix : projet docix"),
            "Déploiement Docix"
        );
    }
}
