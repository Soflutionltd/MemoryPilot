//! French language retrieval benchmark.
//!
//! Builds an in-memory database, seeds a curated French corpus (notes,
//! decisions, bug reports, dialog transcripts) and replays a set of
//! French queries to measure R@1, R@5, R@10, NDCG@10 and MRR on a closed
//! reproducible dataset. The goal is to detect any local-only retrieval
//! regression on French content, since the LongMemEval benchmark is
//! English-only.
//!
//! Usage from CLI:
//!     memorypilot --benchmark-fr [--min-r5 PCT]

use rusqlite::Connection;
use serde_json::{json, Value};

use super::{Database, MemoryScope};

struct FrScenario {
    id: &'static str,
    project: &'static str,
    kind: &'static str,
    tags: &'static [&'static str],
    content: &'static str,
}

struct FrQuery {
    id: &'static str,
    query: &'static str,
    project: Option<&'static str>,
    target_id: &'static str,
}

const SCENARIOS: &[FrScenario] = &[
    FrScenario {
        id: "rust-tokio-deadlock",
        project: "memorypilot",
        kind: "decision",
        tags: &["rust", "tokio", "concurrence"],
        content: "Décision technique. On évite désormais d'imbriquer deux Mutex tokio dans la même fonction async, ça crée des deadlocks impossibles à debugger en production. Préférer un seul lock à grain plus large ou un canal mpsc.",
    },
    FrScenario {
        id: "supabase-rls-policy-tenant",
        project: "notegenius",
        kind: "decision",
        tags: &["supabase", "rls", "sécurité"],
        content: "Politique RLS: chaque table multi-tenant doit avoir une policy qui filtre par tenant_id = auth.jwt() ->> 'tenant_id'. On a oublié sur la table notes et un client a vu les notes d'un autre tenant pendant 3 heures.",
    },
    FrScenario {
        id: "stripe-webhook-signature",
        project: "notegenius",
        kind: "bug",
        tags: &["stripe", "webhook", "signature"],
        content: "Bug: la vérification de signature Stripe échouait parce qu'on lisait le body après JSON.parse au lieu du raw body brut. Corrigé en utilisant request.text() avant tout parsing dans l'endpoint webhook.",
    },
    FrScenario {
        id: "svelte-runes-derived",
        project: "notegenius",
        kind: "note",
        tags: &["svelte", "runes", "réactif"],
        content: "Astuce Svelte 5: $derived est paresseux et caché, ne pas mettre d'effets de bord dedans. Pour une dérivation avec side effect, utiliser $effect avec une assignation à un $state interne.",
    },
    FrScenario {
        id: "cloudflare-pages-secret",
        project: "notegenius",
        kind: "fact",
        tags: &["cloudflare", "pages", "secret"],
        content: "Pour ajouter une variable secrète sur Cloudflare Pages: npx wrangler pages secret put NOM_VARIABLE --project-name notegenius. Ne JAMAIS commit la valeur dans .env.local non plus, ça finit dans Git.",
    },
    FrScenario {
        id: "ios-testflight-export",
        project: "planify",
        kind: "fact",
        tags: &["ios", "testflight", "xcode"],
        content: "Pour exporter sur TestFlight depuis Xcode: Product > Archive, puis Distribute App > App Store Connect > Upload. Vérifier que le build number est incrémenté sinon Apple refuse la soumission.",
    },
    FrScenario {
        id: "flutter-instagram-cache",
        project: "planify",
        kind: "decision",
        tags: &["flutter", "instagram", "cache"],
        content: "Décision: on cache les médias Instagram en local via shared_preferences avec une TTL de 6 heures. Au-delà, refresh forcé via l'API. Ça réduit les appels Graph API de 80% et évite les rate limits.",
    },
    FrScenario {
        id: "postgres-jsonb-index",
        project: "memorypilot",
        kind: "note",
        tags: &["postgres", "jsonb", "index"],
        content: "PostgreSQL: pour des recherches fréquentes sur un champ jsonb, créer un index GIN sur jsonb_path_ops. C'est 10x plus rapide qu'un index B-tree classique sur des opérations de containment @>.",
    },
    FrScenario {
        id: "tailwind-design-system",
        project: "notegenius",
        kind: "pattern",
        tags: &["tailwind", "design", "ui"],
        content: "Pattern UI: cards avec rounded-2xl shadow-sm p-6, buttons avec rounded-xl px-6 py-3, et toujours hover:-translate-y-1 transition-all duration-200 sur les éléments interactifs.",
    },
    FrScenario {
        id: "embedding-quantization-int8",
        project: "memorypilot",
        kind: "decision",
        tags: &["embeddings", "quantization", "perf"],
        content: "Décision archi: les embeddings 384-dim float32 sont quantifiés en int8 au stockage, soit 388 octets par vecteur au lieu de 1536. La perte de précision est négligeable et ça permet de tenir 4x plus en RAM.",
    },
    FrScenario {
        id: "knowledge-graph-extraction",
        project: "memorypilot",
        kind: "pattern",
        tags: &["graph", "entités", "extraction"],
        content: "Pattern d'extraction d'entités: regex pour fichiers (chemins .rs .ts .py), tags Markdown, et capitalisations consécutives pour personnes/projets. Pas de LLM, tout local et déterministe.",
    },
    FrScenario {
        id: "git-commit-conventions",
        project: "memorypilot",
        kind: "fact",
        tags: &["git", "commit", "conventions"],
        content: "Convention commits: format type(scope): description, types autorisés feat/fix/refactor/docs/test/chore, un commit par changement logique. Pas de WIP en main, squash avant merge.",
    },
    FrScenario {
        id: "fastembed-multilingual-model",
        project: "memorypilot",
        kind: "decision",
        tags: &["fastembed", "modèle", "multilingue"],
        content: "Choix modèle embeddings: multilingual-e5-small via fastembed, 384 dimensions, ~120 Mo de taille modèle, gère 100+ langues dont français, allemand, espagnol. Performant en local sur CPU.",
    },
    FrScenario {
        id: "fts5-bm25-tuning",
        project: "memorypilot",
        kind: "decision",
        tags: &["fts5", "bm25", "ranking"],
        content: "Tuning BM25 dans FTS5: poids contenu 8.0, tags 5.0, kind 2.0, projet 4.0. Le contenu domine, mais les tags et le projet ajoutent un signal structurel utile pour départager des résultats proches.",
    },
    FrScenario {
        id: "rrf-fusion-k40",
        project: "memorypilot",
        kind: "decision",
        tags: &["rrf", "fusion", "ranking"],
        content: "Fusion RRF k=40 au lieu du k=60 standard. Plus discriminant sur les tops, ce qui correspond mieux à un usage MCP où on demande typiquement les 5 à 10 meilleures memories.",
    },
    FrScenario {
        id: "ann-hnsw-usearch",
        project: "memorypilot",
        kind: "decision",
        tags: &["ann", "hnsw", "usearch"],
        content: "Index ANN local via usearch (HNSW), persisté sur disque, warm-up asynchrone au démarrage. À partir de 5000 memories, le scan vectoriel est court-circuité par les top-K ANN unis aux hits BM25.",
    },
    FrScenario {
        id: "actr-activation-boost",
        project: "memorypilot",
        kind: "pattern",
        tags: &["actr", "cognitif", "ranking"],
        content: "Activation cognitive style ACT-R: les memories accédées récemment et fréquemment reçoivent un boost avant le rerank final. Modélise la mémoire humaine où ce qu'on a utilisé récemment remonte plus vite.",
    },
    FrScenario {
        id: "session-fusion-diversity",
        project: "memorypilot",
        kind: "pattern",
        tags: &["session", "diversité", "ranking"],
        content: "Late fusion par session: on garde diversité en limitant à 2 résultats par session_id dans le top 10. Évite que tout le top vienne d'une seule longue session de chat.",
    },
    FrScenario {
        id: "tree-sitter-code-chunking",
        project: "memorypilot",
        kind: "decision",
        tags: &["tree-sitter", "chunking", "code"],
        content: "Découpage de code source: tree-sitter coupe par unités sémantiques (fonctions, classes, blocs Svelte <script>) plutôt qu'aveuglément par lignes. Préserve la cohérence syntaxique des chunks.",
    },
    FrScenario {
        id: "working-memory-scoped",
        project: "memorypilot",
        kind: "decision",
        tags: &["mémoire", "session", "working"],
        content: "Working memory scopée: stockage in-process éphémère pour scratchpad de session, isolé par session_id et thread_id. Disparait au shutdown, ne pollue jamais le SQLite long-terme.",
    },
    FrScenario {
        id: "duplicate-merge-strategy",
        project: "memorypilot",
        kind: "decision",
        tags: &["dédup", "merge", "stockage"],
        content: "Détection de doublons: similarité Jaccard sur trigrammes >= 0.85 + même project + même kind = merge. Le contenu le plus long gagne, l'importance prend le max, les tags sont unionés.",
    },
    FrScenario {
        id: "ttl-garbage-collection",
        project: "memorypilot",
        kind: "fact",
        tags: &["gc", "ttl", "cleanup"],
        content: "Garbage collection: les memories avec expires_at dépassé sont supprimées en arrière-plan, max une fois par minute. Évite des cleanups massifs synchrones qui bloquent les writes.",
    },
    FrScenario {
        id: "kg-temporal-validity",
        project: "memorypilot",
        kind: "pattern",
        tags: &["graph", "temporel", "kg"],
        content: "Faits du knowledge graph: chaque triplet a un valid_from et un valid_to. Permet de modéliser que 'Antoine travaille chez X' était vrai entre 2020 et 2023, sans perdre l'historique.",
    },
    FrScenario {
        id: "mcp-tool-naming",
        project: "memorypilot",
        kind: "fact",
        tags: &["mcp", "tools", "naming"],
        content: "Nommage outils MCP: snake_case obligatoire, verbe à l'infinitif (search_memory, add_memory, recall). Pas de point ni de slash, le standard MCP les rejette.",
    },
    FrScenario {
        id: "embed-cache-two-tier",
        project: "memorypilot",
        kind: "decision",
        tags: &["cache", "embeddings", "perf"],
        content: "Cache embeddings deux niveaux: LRU 256 entrées en RAM intra-process, plus un cache SQLite write-through 8192 entrées sur disque. Les queries répétées entre sessions sont 4x plus rapides.",
    },
    FrScenario {
        id: "ann-warmup-async",
        project: "memorypilot",
        kind: "decision",
        tags: &["ann", "startup", "perf"],
        content: "Warm-up de l'index ANN au démarrage: déplacé dans un thread détaché. open_at() rend la main en moins de 5 ms même sur 10000 memories, le scan SQL fait le fallback transparent pendant l'hydratation.",
    },
    FrScenario {
        id: "fts5-phrase-near-prefix",
        project: "memorypilot",
        kind: "pattern",
        tags: &["fts5", "phrase", "near"],
        content: "Variantes FTS5 lancées en parallèle: requête prefix*, phrase exacte entre guillemets, et NEAR(terme1 terme2, 5) pour la proximité. Capture les symboles de code, erreurs et concepts nommés.",
    },
    FrScenario {
        id: "longmemeval-r5-baseline",
        project: "memorypilot",
        kind: "fact",
        tags: &["benchmark", "longmemeval", "métrique"],
        content: "Baseline LongMemEval-S: R@5 98.7%, R@10 99.6%, NDCG@10 95.2%, MRR 93.7%. Garde le guard --min-r5 98.5 pour bloquer toute régression dans la pipeline CI.",
    },
    FrScenario {
        id: "credentials-safe-mode",
        project: "memorypilot",
        kind: "decision",
        tags: &["sécurité", "credentials", "recall"],
        content: "Mode safe par défaut: les memories contenant des patterns de credentials (clés API, tokens, mots de passe) sont exclues du recall sauf si mode=full explicite. Évite les fuites accidentelles.",
    },
    FrScenario {
        id: "scope-session-thread-window",
        project: "memorypilot",
        kind: "decision",
        tags: &["scope", "session", "thread"],
        content: "Scope hiérarchique: session_id (la plus large) > thread_id > window_id. Permet d'isoler le scratchpad d'un panel particulier sans polluer les autres conversations en parallèle.",
    },
];

const QUERIES: &[FrQuery] = &[
    FrQuery {
        id: "q-tokio-mutex",
        query: "deadlock entre deux mutex tokio dans une fonction async",
        project: Some("memorypilot"),
        target_id: "rust-tokio-deadlock",
    },
    FrQuery {
        id: "q-rls-tenant",
        query: "fuite de données entre tenants à cause de RLS manquante",
        project: Some("notegenius"),
        target_id: "supabase-rls-policy-tenant",
    },
    FrQuery {
        id: "q-stripe-sig",
        query: "signature stripe invalide raw body parsé trop tôt",
        project: Some("notegenius"),
        target_id: "stripe-webhook-signature",
    },
    FrQuery {
        id: "q-derived-effet",
        query: "comment faire un effet de bord avec une dérivation Svelte 5",
        project: Some("notegenius"),
        target_id: "svelte-runes-derived",
    },
    FrQuery {
        id: "q-secret-cf",
        query: "ajouter un secret sur Cloudflare Pages en ligne de commande",
        project: Some("notegenius"),
        target_id: "cloudflare-pages-secret",
    },
    FrQuery {
        id: "q-testflight",
        query: "déposer un build iOS sur TestFlight via Xcode",
        project: Some("planify"),
        target_id: "ios-testflight-export",
    },
    FrQuery {
        id: "q-cache-instagram",
        query: "stratégie de cache pour les médias Instagram dans Flutter",
        project: Some("planify"),
        target_id: "flutter-instagram-cache",
    },
    FrQuery {
        id: "q-jsonb-perf",
        query: "accélérer les requêtes containment sur du jsonb postgres",
        project: Some("memorypilot"),
        target_id: "postgres-jsonb-index",
    },
    FrQuery {
        id: "q-cards-style",
        query: "style des cards et boutons dans le design system Tailwind",
        project: Some("notegenius"),
        target_id: "tailwind-design-system",
    },
    FrQuery {
        id: "q-int8-vectors",
        query: "compresser les embeddings en int8 pour réduire la RAM",
        project: Some("memorypilot"),
        target_id: "embedding-quantization-int8",
    },
    FrQuery {
        id: "q-graph-extract",
        query: "extraire entités sans appeler de modèle LLM",
        project: Some("memorypilot"),
        target_id: "knowledge-graph-extraction",
    },
    FrQuery {
        id: "q-commits",
        query: "format des messages de commit conventionnels",
        project: Some("memorypilot"),
        target_id: "git-commit-conventions",
    },
    FrQuery {
        id: "q-modele-fastembed",
        query: "quel modèle d'embedding multilingue local utiliser",
        project: Some("memorypilot"),
        target_id: "fastembed-multilingual-model",
    },
    FrQuery {
        id: "q-bm25-poids",
        query: "comment pondérer les colonnes dans BM25 FTS5",
        project: Some("memorypilot"),
        target_id: "fts5-bm25-tuning",
    },
    FrQuery {
        id: "q-rrf-k",
        query: "valeur du paramètre k dans la fusion RRF",
        project: Some("memorypilot"),
        target_id: "rrf-fusion-k40",
    },
    FrQuery {
        id: "q-hnsw-local",
        query: "index ANN HNSW local pour scaling vectoriel",
        project: Some("memorypilot"),
        target_id: "ann-hnsw-usearch",
    },
    FrQuery {
        id: "q-actr",
        query: "boost des memories récemment et fréquemment utilisées",
        project: Some("memorypilot"),
        target_id: "actr-activation-boost",
    },
    FrQuery {
        id: "q-session-diversite",
        query: "diversifier les résultats par session pour éviter la concentration",
        project: Some("memorypilot"),
        target_id: "session-fusion-diversity",
    },
    FrQuery {
        id: "q-treesitter",
        query: "découpage sémantique du code source par tree-sitter",
        project: Some("memorypilot"),
        target_id: "tree-sitter-code-chunking",
    },
    FrQuery {
        id: "q-working-mem",
        query: "scratchpad éphémère par session sans pollution long-terme",
        project: Some("memorypilot"),
        target_id: "working-memory-scoped",
    },
    FrQuery {
        id: "q-dedup",
        query: "stratégie de fusion des doublons dans le stockage",
        project: Some("memorypilot"),
        target_id: "duplicate-merge-strategy",
    },
    FrQuery {
        id: "q-gc-ttl",
        query: "nettoyage des memories expirées en arrière plan",
        project: Some("memorypilot"),
        target_id: "ttl-garbage-collection",
    },
    FrQuery {
        id: "q-kg-temporel",
        query: "validité temporelle des faits dans le knowledge graph",
        project: Some("memorypilot"),
        target_id: "kg-temporal-validity",
    },
    FrQuery {
        id: "q-mcp-naming",
        query: "convention de nommage des outils MCP",
        project: Some("memorypilot"),
        target_id: "mcp-tool-naming",
    },
    FrQuery {
        id: "q-cache-2tier",
        query: "cache deux niveaux RAM et disque pour embeddings de query",
        project: Some("memorypilot"),
        target_id: "embed-cache-two-tier",
    },
    FrQuery {
        id: "q-warmup-async",
        query: "démarrage non bloquant avec warm-up ANN en arrière plan",
        project: Some("memorypilot"),
        target_id: "ann-warmup-async",
    },
    FrQuery {
        id: "q-fts-variants",
        query: "requêtes FTS5 phrase prefix et proximité NEAR",
        project: Some("memorypilot"),
        target_id: "fts5-phrase-near-prefix",
    },
    FrQuery {
        id: "q-baseline-lme",
        query: "scores de référence sur le benchmark LongMemEval",
        project: Some("memorypilot"),
        target_id: "longmemeval-r5-baseline",
    },
    FrQuery {
        id: "q-safe-mode",
        query: "mode sûr qui exclut les credentials du recall par défaut",
        project: Some("memorypilot"),
        target_id: "credentials-safe-mode",
    },
    FrQuery {
        id: "q-scope",
        query: "hiérarchie des scopes session thread window",
        project: Some("memorypilot"),
        target_id: "scope-session-thread-window",
    },
];

impl Database {
    /// Run the French language retrieval benchmark on a clean temporary
    /// dataset and return aggregated metrics.
    pub fn benchmark_fr() -> Result<Value, String> {
        let tmp_dir = std::env::temp_dir().join(format!(
            "memorypilot-fr-bench-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&tmp_dir)
            .map_err(|error| format!("FR bench: create tmp dir: {}", error))?;
        let db_path = tmp_dir.join("bench.db");
        let result = run_fr_bench(&db_path);
        let _ = std::fs::remove_dir_all(&tmp_dir);
        result
    }
}

fn run_fr_bench(db_path: &std::path::Path) -> Result<Value, String> {
    let db = Database::open_at(db_path)?;
    let scope = MemoryScope::default();
    let mut id_map = std::collections::HashMap::new();

    for spec in SCENARIOS {
        let (memory, _) = db.add_memory(
            spec.content,
            spec.kind,
            Some(spec.project),
            &spec
                .tags
                .iter()
                .map(|tag| tag.to_string())
                .collect::<Vec<_>>(),
            "bench-fr",
            3,
            None,
            None,
            &scope,
        )?;
        id_map.insert(spec.id.to_string(), memory.id);
    }

    wait_for_embeddings(db_path)?;

    let mut hit_at_1 = 0usize;
    let mut hit_at_5 = 0usize;
    let mut hit_at_10 = 0usize;
    let mut reciprocal_ranks = Vec::with_capacity(QUERIES.len());
    let mut ndcgs = Vec::with_capacity(QUERIES.len());
    let mut total_search_ms = 0.0f64;
    let mut per_query_results = Vec::with_capacity(QUERIES.len());

    for query in QUERIES {
        let target_db_id = id_map
            .get(query.target_id)
            .ok_or_else(|| format!("FR bench: missing target id {}", query.target_id))?
            .clone();

        let started = std::time::Instant::now();
        let results = db.search(query.query, 10, query.project, None, None, None)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        total_search_ms += elapsed_ms;

        let position = results
            .iter()
            .position(|result| result.memory.id == target_db_id);
        let rank = position.map(|index| index + 1);

        if let Some(rank) = rank {
            if rank <= 1 {
                hit_at_1 += 1;
            }
            if rank <= 5 {
                hit_at_5 += 1;
            }
            if rank <= 10 {
                hit_at_10 += 1;
            }
            reciprocal_ranks.push(1.0 / rank as f64);
            let dcg = 1.0 / ((rank as f64) + 1.0).log2();
            ndcgs.push(dcg);
        } else {
            reciprocal_ranks.push(0.0);
            ndcgs.push(0.0);
        }

        per_query_results.push(json!({
            "query_id": query.id,
            "query": query.query,
            "target": query.target_id,
            "rank": rank,
            "search_ms": format!("{:.2}", elapsed_ms),
        }));
    }

    let n = QUERIES.len() as f64;
    let mrr = reciprocal_ranks.iter().sum::<f64>() / n;
    let ndcg10 = ndcgs.iter().sum::<f64>() / n;
    let avg_search_ms = total_search_ms / n;

    Ok(json!({
        "dataset": {
            "name": "memorypilot-fr-30",
            "memories": SCENARIOS.len(),
            "queries": QUERIES.len(),
            "language": "fr",
        },
        "metrics": {
            "recall_at_1": format!("{:.1}%", hit_at_1 as f64 / n * 100.0),
            "recall_at_5": format!("{:.1}%", hit_at_5 as f64 / n * 100.0),
            "recall_at_10": format!("{:.1}%", hit_at_10 as f64 / n * 100.0),
            "ndcg_at_10": format!("{:.1}%", ndcg10 * 100.0),
            "mrr": format!("{:.1}%", mrr * 100.0),
            "avg_search_ms": format!("{:.2}", avg_search_ms),
        },
        "search_engine": "BM25 + cosine RRF (k=40)",
        "per_query": per_query_results,
    }))
}

fn wait_for_embeddings(db_path: &std::path::Path) -> Result<(), String> {
    let started = std::time::Instant::now();
    loop {
        let conn = Connection::open(db_path)
            .map_err(|error| format!("FR bench wait probe: {}", error))?;
        let _ = conn.busy_timeout(std::time::Duration::from_secs(2));
        let pending: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NULL",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        if pending == 0 {
            return Ok(());
        }
        if started.elapsed().as_secs() > 120 {
            return Err(format!(
                "FR bench: timeout waiting for embeddings ({} pending)",
                pending
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}
