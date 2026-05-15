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
    // --- Extended dataset (v2): broader vocabulary, more domains, more
    // conversational and bug-report style memories so the benchmark
    // exercises cross-encoder rerank under realistic French usage.
    FrScenario {
        id: "rust-arc-mutex-vs-rwlock",
        project: "memorypilot",
        kind: "decision",
        tags: &["rust", "concurrence", "lock"],
        content: "Choix Arc<Mutex> vs Arc<RwLock>: utiliser RwLock seulement si lectures >> écritures (typique pour caches en lecture). Pour 50/50 reads/writes Mutex est souvent plus rapide à cause du coût de réveil des lecteurs.",
    },
    FrScenario {
        id: "tokio-spawn-vs-spawn-blocking",
        project: "memorypilot",
        kind: "pattern",
        tags: &["tokio", "blocking", "async"],
        content: "tokio::spawn pour des futures purement async, tokio::task::spawn_blocking pour du code CPU-bound ou un appel synchrone (rusqlite, lecture fichier sync). Mélanger les deux dans le même runtime tue les performances.",
    },
    FrScenario {
        id: "rust-error-thiserror-anyhow",
        project: "memorypilot",
        kind: "decision",
        tags: &["rust", "erreurs", "thiserror"],
        content: "Convention erreurs Rust: thiserror dans les bibliothèques (erreurs typées, exposées dans l'API publique), anyhow dans les binaires (chaînes d'erreur opaques avec contexte). Ne jamais mélanger les deux dans une crate publique.",
    },
    FrScenario {
        id: "sqlite-wal-checkpoint",
        project: "memorypilot",
        kind: "fact",
        tags: &["sqlite", "wal", "checkpoint"],
        content: "SQLite WAL grossit indéfiniment si aucun checkpoint n'est lancé. Configurer wal_autocheckpoint à 1000 pages, et lancer manuellement PRAGMA wal_checkpoint(TRUNCATE) en arrière-plan pour libérer l'espace disque.",
    },
    FrScenario {
        id: "supabase-rpc-vs-rest",
        project: "notegenius",
        kind: "pattern",
        tags: &["supabase", "rpc", "api"],
        content: "Préférer Supabase RPC (fonctions PL/pgSQL exposées) plutôt que REST quand on a besoin de transactions multi-tables ou de logique métier serveur. REST suffit pour CRUD simple, RPC pour les opérations atomiques.",
    },
    FrScenario {
        id: "supabase-realtime-postgres-changes",
        project: "notegenius",
        kind: "fact",
        tags: &["supabase", "realtime", "postgres"],
        content: "Supabase Realtime: pour suivre les modifications d'une table, créer un canal sur 'postgres_changes' avec event INSERT/UPDATE/DELETE, schema public, table cible. Ne pas oublier d'activer la replication sur la table dans le dashboard.",
    },
    FrScenario {
        id: "stripe-idempotency-key",
        project: "notegenius",
        kind: "pattern",
        tags: &["stripe", "idempotence", "api"],
        content: "Toujours envoyer un Idempotency-Key sur les requêtes POST Stripe sensibles (création de paiement, abonnement). Sinon un retry réseau peut déclencher un double prélèvement. Utiliser un UUID v4 par tentative logique.",
    },
    FrScenario {
        id: "stripe-checkout-vs-elements",
        project: "notegenius",
        kind: "decision",
        tags: &["stripe", "checkout", "ui"],
        content: "Stripe Checkout (page hébergée) pour la majorité des cas: gestion 3DS, taxes, codes promo automatique. Stripe Elements seulement si le branding est critique au point de coder son propre formulaire de carte avec PCI scope.",
    },
    FrScenario {
        id: "svelte-effect-cleanup",
        project: "notegenius",
        kind: "pattern",
        tags: &["svelte", "runes", "effect"],
        content: "Pattern Svelte 5: dans un $effect, retourner une fonction de cleanup pour annuler abonnements ou timers. La fonction est appelée avant chaque réexécution et au teardown du composant. Évite les fuites de listeners.",
    },
    FrScenario {
        id: "svelte-load-vs-onmount",
        project: "notegenius",
        kind: "decision",
        tags: &["svelte", "kit", "load"],
        content: "Dans SvelteKit: charger les données via load() en +page.ts (server ou universal) plutôt que dans onMount/$effect. Permet le SSR, le streaming, et évite le flash sans contenu initial. onMount uniquement pour les API navigateur pures.",
    },
    FrScenario {
        id: "tailwind-v4-css-only",
        project: "notegenius",
        kind: "fact",
        tags: &["tailwind", "v4", "css"],
        content: "Tailwind v4: configuration en CSS pur via @theme, plus besoin de tailwind.config.js. @theme inline pour les variables locales, @theme pour les variables globales. Les anciens plugins JavaScript ne sont plus supportés tels quels.",
    },
    FrScenario {
        id: "cloudflare-workers-vs-pages",
        project: "notegenius",
        kind: "decision",
        tags: &["cloudflare", "workers", "pages"],
        content: "Cloudflare Pages pour sites statiques + functions serverless avec déploiement git auto. Workers pour API custom complexes ou besoins de Durable Objects, KV, Queues. Pages utilise Workers en interne pour ses functions.",
    },
    FrScenario {
        id: "cloudflare-durable-objects",
        project: "notegenius",
        kind: "fact",
        tags: &["cloudflare", "durable-objects", "stateful"],
        content: "Durable Objects: instance unique globale par ID, stockage transactionnel, idéal pour rooms de chat, locks distribués, compteurs cohérents. Coût: latence légèrement plus élevée que KV mais cohérence forte garantie.",
    },
    FrScenario {
        id: "ios-app-extension-shared-keychain",
        project: "planify",
        kind: "fact",
        tags: &["ios", "extension", "keychain"],
        content: "Pour partager des secrets entre app principale et widget iOS, utiliser un keychain access group commun configuré dans les capabilities Xcode. Les fichiers UserDefaults peuvent passer par App Groups au lieu du keychain pour les non-secrets.",
    },
    FrScenario {
        id: "ios-background-task-bgprocessing",
        project: "planify",
        kind: "pattern",
        tags: &["ios", "background", "bgtask"],
        content: "iOS: BGProcessingTaskRequest pour tâches lourdes (sync, ML inference) avec besoin de courant et wifi. BGAppRefreshTask pour rafraîchissements rapides de contenu. Toujours enregistrer les identifiants dans Info.plist sous BGTaskSchedulerPermittedIdentifiers.",
    },
    FrScenario {
        id: "android-jetpack-compose-state",
        project: "planify",
        kind: "pattern",
        tags: &["android", "compose", "state"],
        content: "Jetpack Compose: remember { mutableStateOf } pour état local, hoisting via paramètres pour partage parent-enfant, ViewModel pour état partagé entre écrans. Ne JAMAIS utiliser un objet mutable global comme état de Compose, ça casse la recomposition.",
    },
    FrScenario {
        id: "flutter-riverpod-vs-bloc",
        project: "planify",
        kind: "decision",
        tags: &["flutter", "riverpod", "bloc"],
        content: "Choix de state management Flutter: Riverpod pour nouvelle codebase (compile-time safety, providers composables, testable trivialement). Bloc reste pertinent si l'équipe est déjà formée et qu'on veut une séparation stricte UI/logique.",
    },
    FrScenario {
        id: "postgres-explain-analyze",
        project: "memorypilot",
        kind: "pattern",
        tags: &["postgres", "explain", "perf"],
        content: "Pour debug une requête lente PostgreSQL: EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON). Chercher les Seq Scan sur grosses tables, les Hash Join avec spill disque, les sort_method=external. Buffers = lectures disque vs cache.",
    },
    FrScenario {
        id: "postgres-pgvector-cosine",
        project: "memorypilot",
        kind: "fact",
        tags: &["postgres", "pgvector", "vector"],
        content: "pgvector: opérateur <=> pour distance cosinus, <-> pour L2, <#> pour produit scalaire négatif. Index ivfflat pour grands datasets (10k+), hnsw pour recall maximal. Toujours analyser après bulk load avant de créer l'index.",
    },
    FrScenario {
        id: "redis-stream-vs-pubsub",
        project: "memorypilot",
        kind: "decision",
        tags: &["redis", "stream", "pubsub"],
        content: "Redis Streams pour bus d'événements persistant avec consumer groups (replay possible, ack par message). Pub/Sub pour notifications fire-and-forget (perdues si aucun subscriber). Stream est presque toujours le bon choix sauf pour push UI temps réel.",
    },
    FrScenario {
        id: "git-rebase-vs-merge",
        project: "memorypilot",
        kind: "decision",
        tags: &["git", "rebase", "merge"],
        content: "Workflow git: rebase sur sa propre branche feature avant push (historique linéaire), merge avec --no-ff pour intégrer dans main (préserve la trace de la PR). Jamais rebase sur une branche partagée déjà poussée.",
    },
    FrScenario {
        id: "github-actions-cache-key",
        project: "memorypilot",
        kind: "pattern",
        tags: &["github", "ci", "cache"],
        content: "GitHub Actions cache: la clé doit inclure un hash du lockfile (Cargo.lock, package-lock.json) pour invalider quand les deps changent. Restore-keys avec préfixe partiel permet un fallback partiel si la clé exacte n'existe pas encore.",
    },
    FrScenario {
        id: "docker-multistage-rust",
        project: "memorypilot",
        kind: "pattern",
        tags: &["docker", "rust", "multistage"],
        content: "Dockerfile Rust: stage 1 cargo chef pour cacher les deps en couche séparée, stage 2 cargo build --release, stage 3 image finale gcr.io/distroless/cc avec uniquement le binaire. Réduit l'image de 2 Go à 30 Mo.",
    },
    FrScenario {
        id: "kubernetes-liveness-vs-readiness",
        project: "memorypilot",
        kind: "fact",
        tags: &["k8s", "probe", "health"],
        content: "Kubernetes probes: liveness redémarre le pod si le check échoue (à utiliser pour deadlocks). Readiness retire le pod du load balancer (utile pendant warm-up ou maintenance). Échec readiness ne tue pas le pod, échec liveness oui.",
    },
    FrScenario {
        id: "openai-streaming-sse",
        project: "memorypilot",
        kind: "pattern",
        tags: &["openai", "sse", "streaming"],
        content: "OpenAI streaming: reader sur le body fetch, parser ligne par ligne les chunks 'data: {...}\\n\\n'. Le dernier chunk est 'data: [DONE]'. Penser à un keep-alive côté serveur si proxy nginx, sinon timeout après 60 s.",
    },
    FrScenario {
        id: "anthropic-tool-use",
        project: "memorypilot",
        kind: "fact",
        tags: &["anthropic", "tools", "claude"],
        content: "Claude tool use: dans la requête, fournir 'tools' avec name, description et input_schema JSON. La réponse contient stop_reason='tool_use' et un bloc 'tool_use' avec input. Renvoyer ensuite un message 'tool_result' avec tool_use_id pour continuer le tour.",
    },
    FrScenario {
        id: "mcp-stdio-vs-sse",
        project: "memorypilot",
        kind: "decision",
        tags: &["mcp", "stdio", "transport"],
        content: "Transport MCP: stdio pour processus enfant local (Cursor lance le binaire), SSE/HTTP pour serveur distant partagé. stdio évite tout setup réseau et est le choix par défaut. SSE pour multi-clients ou déploiement cloud.",
    },
    FrScenario {
        id: "mcp-resources-vs-tools",
        project: "memorypilot",
        kind: "fact",
        tags: &["mcp", "resources", "tools"],
        content: "MCP: tools pour actions avec effets (write, search), resources pour données lisibles attachées au contexte LLM (fichiers, configs). Les resources sont automatiquement injectées par le client, les tools sont appelés explicitement par le modèle.",
    },
    FrScenario {
        id: "fastembed-batch-size",
        project: "memorypilot",
        kind: "fact",
        tags: &["fastembed", "batch", "perf"],
        content: "fastembed batch optimal: 32 sur CPU pour multilingual-e5-small. Au-delà, la latence par item augmente sans gain de throughput. Sur GPU, monter à 128. Toujours mesurer p95 et pas seulement throughput moyen.",
    },
    FrScenario {
        id: "onnx-runtime-execution-providers",
        project: "memorypilot",
        kind: "decision",
        tags: &["onnx", "runtime", "providers"],
        content: "ONNX Runtime providers: CPUExecutionProvider partout, CoreMLExecutionProvider sur macOS Apple Silicon (gain x2-3 sur petits modèles), CUDAExecutionProvider sur NVIDIA GPU. Toujours fallback CPU explicite, sinon crash silencieux si provider absent.",
    },
    FrScenario {
        id: "vector-db-faiss-vs-usearch",
        project: "memorypilot",
        kind: "decision",
        tags: &["vector", "faiss", "usearch"],
        content: "Choix index vectoriel: usearch pour Rust pur, single-binary, persistance triviale. FAISS pour C++/Python avec besoins exotiques (PQ, IVF tunables). Pour <1M vecteurs, usearch HNSW est plus simple et aussi rapide sur Apple Silicon.",
    },
    FrScenario {
        id: "embeddings-normalize-l2",
        project: "memorypilot",
        kind: "fact",
        tags: &["embeddings", "normalize", "cosine"],
        content: "Embeddings: normaliser L2 systématiquement après fastembed pour que la similarité cosinus se réduise à un produit scalaire. Sans normalisation, le ranking peut être correct mais les seuils numériques deviennent instables.",
    },
    FrScenario {
        id: "rerank-cross-encoder-late",
        project: "memorypilot",
        kind: "pattern",
        tags: &["rerank", "cross-encoder", "ranking"],
        content: "Pipeline retrieval: retrieve top-50 par BM25+vector, ensuite rerank top-12 avec un cross-encoder (jina-reranker-v2 multilingue). Coût ~150 ms par query, gain typique +5-10 pp sur R@5 en non-anglais.",
    },
    FrScenario {
        id: "telemetry-jsonl-shipping",
        project: "memorypilot",
        kind: "decision",
        tags: &["telemetry", "jsonl", "logs"],
        content: "Format des traces de retrieval: JSONL une ligne par recherche, écriture stderr ou fichier configurable via env var. Permet streaming vers vector/loki/elasticsearch sans parsing custom et reste lisible cat-friendly en local.",
    },
    FrScenario {
        id: "mcp-pagination-cursor",
        project: "memorypilot",
        kind: "fact",
        tags: &["mcp", "pagination", "cursor"],
        content: "Pagination MCP: utiliser un cursor opaque renvoyé dans nextCursor, pas un offset numérique. Permet d'évoluer le tri/filtre sans casser les clients en cours de pagination. Le client renvoie le cursor sans le décoder.",
    },
    FrScenario {
        id: "react-server-components-vs-client",
        project: "notegenius",
        kind: "decision",
        tags: &["react", "rsc", "next"],
        content: "React Server Components: par défaut côté serveur (zéro JS au client). Ajouter 'use client' uniquement sur les feuilles interactives (forms, charts). Hisser les RSC le plus haut possible dans l'arbre pour minimiser le bundle.",
    },
    FrScenario {
        id: "websocket-vs-sse-temps-reel",
        project: "notegenius",
        kind: "decision",
        tags: &["websocket", "sse", "temps-reel"],
        content: "WebSocket si bidirectionnel nécessaire (chat avec typing indicators), SSE si unidirectionnel server->client (notifications, updates de statut). SSE plus simple à proxifier, supporte la reconnexion automatique côté navigateur.",
    },
    FrScenario {
        id: "graphql-vs-trpc-typed-rpc",
        project: "notegenius",
        kind: "decision",
        tags: &["graphql", "trpc", "api"],
        content: "tRPC pour monorepo TS où front et back partagent le même typecheck (zéro codegen, type inference end-to-end). GraphQL si plusieurs clients hétérogènes ou besoin d'introspection schema. REST si besoin de cacheable HTTP standard.",
    },
    FrScenario {
        id: "playwright-vs-cypress-e2e",
        project: "notegenius",
        kind: "decision",
        tags: &["playwright", "cypress", "e2e"],
        content: "Playwright pour multi-browser (Chromium, Firefox, WebKit), parallélisme natif, attente intelligente sans flaky waits. Cypress reste plus simple pour tester rapidement un Single Page App moderne mais limité au navigateur unique en pratique.",
    },
    FrScenario {
        id: "vitest-vs-jest-unit",
        project: "notegenius",
        kind: "decision",
        tags: &["vitest", "jest", "test"],
        content: "Vitest: ESM natif, démarrage 10x plus rapide que Jest, compatible API Jest. Choix par défaut pour tout projet Vite/SvelteKit/React moderne. Jest reste pour codebases legacy CommonJS où la migration coûterait plus cher.",
    },
    FrScenario {
        id: "yarn-pnpm-vs-npm",
        project: "notegenius",
        kind: "decision",
        tags: &["pnpm", "yarn", "npm"],
        content: "pnpm: store global content-addressable, links symboliques, économise des Go sur disque dans monorepo. Plus rapide que npm/yarn classic. Workspaces natifs solides. Choix par défaut pour nouveaux projets multi-packages.",
    },
    FrScenario {
        id: "design-tokens-figma-style-dictionary",
        project: "notegenius",
        kind: "pattern",
        tags: &["design", "tokens", "figma"],
        content: "Design tokens: source de vérité dans Figma Variables, export JSON via plugin Tokens Studio, transformation via Style Dictionary vers CSS custom properties / Tailwind config / iOS xcassets. Garde Figma et code synchronisés sans copy-paste manuel.",
    },
    FrScenario {
        id: "accessibilite-aria-live",
        project: "notegenius",
        kind: "fact",
        tags: &["a11y", "aria", "annonces"],
        content: "Accessibilité: aria-live='polite' pour annonces non urgentes (toast, status mis à jour), aria-live='assertive' pour erreurs critiques bloquantes. Toujours sur un container existant au mount, pas créé dynamiquement à chaque annonce.",
    },
    FrScenario {
        id: "i18n-icu-message-format",
        project: "notegenius",
        kind: "pattern",
        tags: &["i18n", "icu", "pluralisation"],
        content: "Internationalisation: utiliser ICU MessageFormat pour la pluralisation et le genre. Pas concaténer 'Vous avez ' + count + ' notes', toujours '{count, plural, one {# note} other {# notes}}'. Indispensable pour FR/RU/PL/AR.",
    },
    FrScenario {
        id: "dark-mode-prefers-color-scheme",
        project: "notegenius",
        kind: "pattern",
        tags: &["dark-mode", "css", "media"],
        content: "Dark mode: respecter prefers-color-scheme par défaut, override manuel via attribut data-theme sur <html>, persistence dans localStorage. Hydrater côté serveur via cookie pour éviter le flash blanc avant le JS.",
    },
    FrScenario {
        id: "csrf-double-submit-cookie",
        project: "notegenius",
        kind: "fact",
        tags: &["sécurité", "csrf", "cookie"],
        content: "Protection CSRF moderne: double submit cookie (token dans cookie SameSite=Lax + même token dans header X-CSRF-Token). Plus simple que les anciens tokens server-side stockés en session, suffisant pour la majorité des SPA.",
    },
    FrScenario {
        id: "jwt-rotation-refresh",
        project: "notegenius",
        kind: "decision",
        tags: &["jwt", "auth", "refresh"],
        content: "Auth: access token JWT court (15 min) + refresh token long (30 j) en cookie httpOnly. Rotation du refresh à chaque usage (token use=once) pour détecter les vols. Si rotation détecte un token déjà consommé, invalider toute la famille.",
    },
    FrScenario {
        id: "passkey-webauthn-replace-password",
        project: "notegenius",
        kind: "decision",
        tags: &["passkey", "webauthn", "auth"],
        content: "Passkeys WebAuthn: remplacent à terme les mots de passe. Implémentation via @simplewebauthn/server et @simplewebauthn/browser. UX: ajouter passkey en plus du mot de passe pendant la transition, ne pas forcer pendant 6 mois.",
    },
    FrScenario {
        id: "rate-limit-token-bucket",
        project: "notegenius",
        kind: "pattern",
        tags: &["rate-limit", "redis", "bucket"],
        content: "Rate limiting: token bucket dans Redis avec script Lua atomique (tokens, last_refill, capacity). Préférable au sliding window pour absorber les pics légitimes tout en bloquant les abus soutenus. ~50 µs par check.",
    },
    FrScenario {
        id: "queue-bullmq-vs-trigger-dev",
        project: "notegenius",
        kind: "decision",
        tags: &["queue", "bullmq", "trigger"],
        content: "Background jobs: BullMQ + Redis pour contrôle total et coût bas. Trigger.dev / Inngest pour DX moderne, retry/replay UI, observabilité incluse mais coût mensuel. Pour un side-project commencer BullMQ, migrer si la complexité explose.",
    },
    FrScenario {
        id: "monorepo-turbo-vs-nx",
        project: "notegenius",
        kind: "decision",
        tags: &["monorepo", "turbo", "nx"],
        content: "Monorepo TS: Turborepo léger, pipeline de cache simple, parfait pour <10 packages. Nx pour grosses orgs avec generators, plugins riches, dependency graph visualisé. Turbo gagne en vélocité d'apprentissage, Nx en outillage.",
    },
    FrScenario {
        id: "sentry-source-maps-upload",
        project: "notegenius",
        kind: "fact",
        tags: &["sentry", "source-maps", "monitoring"],
        content: "Sentry source maps: générer en build avec hidden-source-map (ne pas exposer publiquement), uploader via @sentry/cli en post-build CI avec --release matching la version. Sans ça, les stack traces sont illisibles en prod.",
    },
    FrScenario {
        id: "log-structured-pino",
        project: "notegenius",
        kind: "decision",
        tags: &["logs", "pino", "structured"],
        content: "Logs Node: pino pour log JSON structuré ultra-rapide, transport pino-pretty seulement en dev. JAMAIS de console.log en prod, perdre la structure casse l'agrégation downstream. Niveau par défaut info, debug uniquement avec env var.",
    },
    FrScenario {
        id: "tracing-rust-spans",
        project: "memorypilot",
        kind: "pattern",
        tags: &["rust", "tracing", "spans"],
        content: "Rust tracing: instrumenter avec #[tracing::instrument(skip(big_arg))] sur les fonctions importantes, créer des spans manuels pour les boucles, attacher les fields utiles. tracing-subscriber JSON en prod, fmt humain en dev.",
    },
    FrScenario {
        id: "actix-vs-axum-rust-web",
        project: "memorypilot",
        kind: "decision",
        tags: &["rust", "axum", "actix"],
        content: "Choix framework web Rust: axum pour nouvelles APIs (tower middleware ecosystem, ergonomie tokio native, syntaxe extracteurs claire). Actix-web reste valide pour codebases existantes ou besoins ultra-perfs SIMD spécifiques.",
    },
    FrScenario {
        id: "wasm-target-rust-wasm-pack",
        project: "memorypilot",
        kind: "fact",
        tags: &["wasm", "rust", "wasm-pack"],
        content: "Compiler Rust en WebAssembly: cargo target wasm32-unknown-unknown, wasm-pack build --target web pour bundle ESM. Optimiser avec wasm-opt -O3 (50% taille en moins). Attention au coût d'un crate qui dépend de tokio (incompatible).",
    },
    FrScenario {
        id: "webgpu-compute-vs-webgl",
        project: "memorypilot",
        kind: "fact",
        tags: &["webgpu", "compute", "webgl"],
        content: "WebGPU: API moderne unifiée graphics + compute, supporte compute shaders WGSL. WebGL limité à graphics et bricolages compute via fragment shader. Pour ML inference dans navigateur, WebGPU est la voie standard désormais.",
    },
    FrScenario {
        id: "browser-cache-stale-while-revalidate",
        project: "notegenius",
        kind: "fact",
        tags: &["http", "cache", "swr"],
        content: "Header Cache-Control: stale-while-revalidate=60 sert la version cachée immédiatement et déclenche un refresh en arrière-plan. Excellent pour API en lecture lourde. Combiner avec ETag pour éviter les transferts inutiles.",
    },
    FrScenario {
        id: "image-avif-vs-webp",
        project: "notegenius",
        kind: "fact",
        tags: &["image", "avif", "webp"],
        content: "Format image: AVIF -50% poids vs WebP, supporté Chrome/Firefox/Safari 16+. Servir via <picture> avec sources AVIF, WebP, JPEG fallback. Génération via sharp (Node) ou cavif (Rust). Coûteux à encoder, cacher agressivement.",
    },
    FrScenario {
        id: "ffmpeg-thumbnail-fast",
        project: "notegenius",
        kind: "fact",
        tags: &["ffmpeg", "thumbnail", "video"],
        content: "Génération thumbnail vidéo rapide: ffmpeg -ss 00:00:01 AVANT -i input.mp4 (seek rapide imprécis). Pour précision, mettre -ss APRÈS -i (lent mais frame-accurate). Toujours -frames:v 1 et -q:v 2 pour qualité jpg.",
    },
    FrScenario {
        id: "imagemagick-batch-resize",
        project: "notegenius",
        kind: "pattern",
        tags: &["imagemagick", "batch", "resize"],
        content: "Resize batch ImageMagick: mogrify -resize 1024x> -quality 85 -strip *.jpg. Le > limite à shrink only (pas d'upscale). -strip enlève les métadonnées EXIF (gain 10-30%). mogrify modifie en place, convert pour nouveau fichier.",
    },
    FrScenario {
        id: "supabase-edge-functions-deno",
        project: "notegenius",
        kind: "fact",
        tags: &["supabase", "edge", "deno"],
        content: "Supabase Edge Functions: runtime Deno, déployées sur le réseau global. Idéales pour webhooks, intégrations API tierces avec secrets stockés via supabase secrets set. Limite cold start ~50 ms, plus rapide qu'AWS Lambda Node.",
    },
    FrScenario {
        id: "cloudflare-r2-vs-s3",
        project: "notegenius",
        kind: "decision",
        tags: &["r2", "s3", "storage"],
        content: "Cloudflare R2: zéro frais de sortie, API S3-compatible, idéal pour assets servis publiquement (CDN gratuit). AWS S3 si on est déjà profondément dans l'écosystème AWS ou besoin spécifique de Lifecycle Rules avancées.",
    },
    FrScenario {
        id: "dns-caa-record-letsencrypt",
        project: "notegenius",
        kind: "fact",
        tags: &["dns", "caa", "letsencrypt"],
        content: "Record DNS CAA: 'example.com. CAA 0 issue \"letsencrypt.org\"'. Restreint quelles autorités peuvent émettre des certs pour le domaine. Empêche un attaquant ayant pris le contrôle d'un sous-domaine d'obtenir un cert wildcard.",
    },
    FrScenario {
        id: "smtp-spf-dkim-dmarc",
        project: "notegenius",
        kind: "fact",
        tags: &["email", "spf", "dkim", "dmarc"],
        content: "Délivrabilité email: configurer SPF (autoriser IPs envoi), DKIM (signer cryptographiquement les emails), DMARC (politique d'alignement et reporting). Sans les trois, Gmail/Outlook taggent en spam ou rejettent silencieusement.",
    },
    FrScenario {
        id: "postmark-resend-vs-sendgrid",
        project: "notegenius",
        kind: "decision",
        tags: &["email", "resend", "postmark"],
        content: "Service email transactionnel: Resend pour DX moderne et React Email templates. Postmark pour fiabilité et stats détaillées. SendGrid décliné en feature mais tarification opaque. Pour <50k mails/mois, Resend est le choix par défaut.",
    },
    FrScenario {
        id: "deeplink-universal-links-android-app-links",
        project: "planify",
        kind: "fact",
        tags: &["mobile", "deeplink", "universal"],
        content: "Deep links: iOS Universal Links via apple-app-site-association servi depuis /.well-known. Android App Links via assetlinks.json. Sans ça, le clic ouvre Safari/Chrome au lieu de l'app. Tester en sortie de TestFlight/internal track.",
    },
    FrScenario {
        id: "push-notifications-fcm-apns",
        project: "planify",
        kind: "fact",
        tags: &["push", "fcm", "apns"],
        content: "Notifications push: utiliser Firebase Cloud Messaging comme couche unifiée iOS+Android. Côté iOS, FCM relaie via APNs (auth p8 key). Token utilisateur invalide à gérer (registration_id_invalid) en supprimant côté serveur sinon erreurs en boucle.",
    },
    FrScenario {
        id: "in-app-purchase-receipt-validation",
        project: "planify",
        kind: "pattern",
        tags: &["iap", "receipt", "validation"],
        content: "In-app purchase: TOUJOURS valider le receipt côté serveur (Apple App Store Server API ou Google Play Developer API). Ne jamais débloquer la fonctionnalité uniquement sur réponse du SDK client, trivial à bypass avec un proxy.",
    },
    FrScenario {
        id: "react-native-vs-flutter",
        project: "planify",
        kind: "decision",
        tags: &["react-native", "flutter", "mobile"],
        content: "React Native vs Flutter: RN si l'équipe est React forte (réutilisation des compétences web). Flutter si UI très custom à pixel près sur les deux plateformes (rendu Skia uniforme). Pour MVP, RN avec Expo accélère time-to-market.",
    },
    FrScenario {
        id: "cli-clap-derive-rust",
        project: "memorypilot",
        kind: "pattern",
        tags: &["rust", "clap", "cli"],
        content: "Rust CLI: clap avec derive macros, struct opts par sous-commande, #[command(name, version)] pour le binaire. clap_complete pour générer auto les complétions bash/zsh/fish. À combiner avec indicatif pour progress bars.",
    },
    FrScenario {
        id: "tui-ratatui-rust",
        project: "memorypilot",
        kind: "fact",
        tags: &["rust", "tui", "ratatui"],
        content: "Interfaces terminal Rust: ratatui (fork actif de tui-rs) pour widgets layouts, crossterm pour gestion événements clavier/souris cross-platform. Architecture event-loop classique. À utiliser pour outils interactifs (TUI), pas pour CLI scripts.",
    },
    FrScenario {
        id: "watchexec-cargo-watch",
        project: "memorypilot",
        kind: "fact",
        tags: &["dev-loop", "watch", "rust"],
        content: "Boucle dev Rust: cargo-watch -x test -x run pour relancer auto à chaque save. Alternative générique watchexec --exts rs cargo run. Sur Apple Silicon, watchexec consomme moins de batterie grâce à kqueue natif.",
    },
    FrScenario {
        id: "criterion-bench-rust",
        project: "memorypilot",
        kind: "fact",
        tags: &["rust", "criterion", "bench"],
        content: "Microbench Rust: criterion pour stats robustes (median, IQR, regression detection), benchmarks dans benches/, lancer avec cargo bench. cargo flamegraph pour profilage CPU. Toujours bench en --release et avec donnees realistes.",
    },
    FrScenario {
        id: "musl-vs-glibc-static",
        project: "memorypilot",
        kind: "decision",
        tags: &["rust", "musl", "static"],
        content: "Build statique Rust: target x86_64-unknown-linux-musl pour binaire portable single-file (parfait pour Docker FROM scratch). Inconvénient: certains crates demandent glibc (e.g. liblzma-sys). Préférer aarch64-apple-darwin pour Mac natif.",
    },
    FrScenario {
        id: "ssh-config-jump-host",
        project: "memorypilot",
        kind: "fact",
        tags: &["ssh", "jump", "config"],
        content: "SSH bastion: dans ~/.ssh/config, 'Host serveur ProxyJump bastion'. Évite -J en CLI à chaque fois. Ajouter ControlMaster auto + ControlPath pour multiplexer plusieurs sessions sans réauthentification.",
    },
    FrScenario {
        id: "vpn-wireguard-vs-tailscale",
        project: "memorypilot",
        kind: "decision",
        tags: &["vpn", "wireguard", "tailscale"],
        content: "VPN équipe: Tailscale pour mesh zero-config (auth via Google/GitHub, ACLs simples). WireGuard pur pour contrôle total avec coordination manuelle des clés. Tailscale gagne pour 95% des cas, WireGuard si compliance ou cost > 100 users.",
    },
    FrScenario {
        id: "1password-vs-bitwarden-team",
        project: "memorypilot",
        kind: "decision",
        tags: &["secrets", "password-manager", "team"],
        content: "Gestion secrets équipe: 1Password Business pour intégration native CLI/SSH agent et Secrets Automation. Bitwarden self-hostable et open source. 1Password gagne sur DX et SSO, Bitwarden sur prix et autonomie.",
    },
    FrScenario {
        id: "doppler-vs-aws-secrets-manager",
        project: "memorypilot",
        kind: "decision",
        tags: &["secrets", "doppler", "aws"],
        content: "Secrets app: Doppler pour multi-cloud avec sync auto vers Vercel/Netlify/Render et CLI développeur. AWS Secrets Manager si déjà tout sur AWS et besoin de KMS rotation auto. Doppler plus simple à démarrer.",
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
    // --- Extended queries (v2): paraphrased and intentionally far
    // from the indexed wording so the bench actually exercises the
    // semantic lane (cross-encoder + embeddings) rather than just
    // BM25 keyword overlap.
    FrQuery {
        id: "q-arc-mutex-rwlock",
        query: "quand préférer RwLock à Mutex en Rust",
        project: Some("memorypilot"),
        target_id: "rust-arc-mutex-vs-rwlock",
    },
    FrQuery {
        id: "q-spawn-blocking",
        query: "exécuter du code synchrone bloquant dans un runtime tokio",
        project: Some("memorypilot"),
        target_id: "tokio-spawn-vs-spawn-blocking",
    },
    FrQuery {
        id: "q-thiserror-anyhow",
        query: "quelle crate d'erreurs choisir entre bibliothèque et binaire",
        project: Some("memorypilot"),
        target_id: "rust-error-thiserror-anyhow",
    },
    FrQuery {
        id: "q-wal-grow",
        query: "fichier WAL SQLite qui grossit sans s'arrêter",
        project: Some("memorypilot"),
        target_id: "sqlite-wal-checkpoint",
    },
    FrQuery {
        id: "q-supabase-rpc",
        query: "appeler une fonction PostgreSQL depuis Supabase",
        project: Some("notegenius"),
        target_id: "supabase-rpc-vs-rest",
    },
    FrQuery {
        id: "q-realtime-supabase",
        query: "écouter les modifications d'une table en temps réel",
        project: Some("notegenius"),
        target_id: "supabase-realtime-postgres-changes",
    },
    FrQuery {
        id: "q-stripe-idem",
        query: "éviter les doubles paiements Stripe sur retry réseau",
        project: Some("notegenius"),
        target_id: "stripe-idempotency-key",
    },
    FrQuery {
        id: "q-checkout-elements",
        query: "page de paiement hébergée Stripe ou formulaire custom",
        project: Some("notegenius"),
        target_id: "stripe-checkout-vs-elements",
    },
    FrQuery {
        id: "q-effect-cleanup",
        query: "nettoyer un abonnement dans un effect Svelte",
        project: Some("notegenius"),
        target_id: "svelte-effect-cleanup",
    },
    FrQuery {
        id: "q-svelte-load",
        query: "où charger les données d'une page SvelteKit",
        project: Some("notegenius"),
        target_id: "svelte-load-vs-onmount",
    },
    FrQuery {
        id: "q-tw-v4-config",
        query: "configurer Tailwind v4 sans fichier JavaScript",
        project: Some("notegenius"),
        target_id: "tailwind-v4-css-only",
    },
    FrQuery {
        id: "q-cf-pages-workers",
        query: "choisir entre Cloudflare Pages et Workers pour une API",
        project: Some("notegenius"),
        target_id: "cloudflare-workers-vs-pages",
    },
    FrQuery {
        id: "q-durable-objects",
        query: "stockage cohérent et stateful sur Cloudflare",
        project: Some("notegenius"),
        target_id: "cloudflare-durable-objects",
    },
    FrQuery {
        id: "q-ios-keychain",
        query: "partager un secret entre l'app et son widget iOS",
        project: Some("planify"),
        target_id: "ios-app-extension-shared-keychain",
    },
    FrQuery {
        id: "q-ios-bgtask",
        query: "déclencher une synchronisation lourde en arrière plan iOS",
        project: Some("planify"),
        target_id: "ios-background-task-bgprocessing",
    },
    FrQuery {
        id: "q-compose-state",
        query: "gérer l'état dans un composable Jetpack Compose",
        project: Some("planify"),
        target_id: "android-jetpack-compose-state",
    },
    FrQuery {
        id: "q-riverpod-bloc",
        query: "Riverpod ou Bloc pour gérer l'état Flutter",
        project: Some("planify"),
        target_id: "flutter-riverpod-vs-bloc",
    },
    FrQuery {
        id: "q-pg-explain",
        query: "diagnostiquer une requête postgres lente",
        project: Some("memorypilot"),
        target_id: "postgres-explain-analyze",
    },
    FrQuery {
        id: "q-pgvector-op",
        query: "opérateurs de distance disponibles dans pgvector",
        project: Some("memorypilot"),
        target_id: "postgres-pgvector-cosine",
    },
    FrQuery {
        id: "q-redis-stream",
        query: "bus d'événements persistant dans Redis avec replay",
        project: Some("memorypilot"),
        target_id: "redis-stream-vs-pubsub",
    },
    FrQuery {
        id: "q-rebase-merge",
        query: "stratégie git pour garder un historique propre",
        project: Some("memorypilot"),
        target_id: "git-rebase-vs-merge",
    },
    FrQuery {
        id: "q-gh-actions-cache",
        query: "invalider le cache GitHub Actions quand les dépendances changent",
        project: Some("memorypilot"),
        target_id: "github-actions-cache-key",
    },
    FrQuery {
        id: "q-docker-rust",
        query: "réduire la taille d'une image Docker pour binaire Rust",
        project: Some("memorypilot"),
        target_id: "docker-multistage-rust",
    },
    FrQuery {
        id: "q-k8s-probe",
        query: "différence entre liveness et readiness Kubernetes",
        project: Some("memorypilot"),
        target_id: "kubernetes-liveness-vs-readiness",
    },
    FrQuery {
        id: "q-openai-stream",
        query: "consommer une réponse streaming OpenAI côté client",
        project: Some("memorypilot"),
        target_id: "openai-streaming-sse",
    },
    FrQuery {
        id: "q-claude-tool",
        query: "appeler une fonction depuis Claude avec tool use",
        project: Some("memorypilot"),
        target_id: "anthropic-tool-use",
    },
    FrQuery {
        id: "q-mcp-transport",
        query: "transport MCP local ou HTTP distant",
        project: Some("memorypilot"),
        target_id: "mcp-stdio-vs-sse",
    },
    FrQuery {
        id: "q-mcp-resources",
        query: "différence entre tool MCP et resource MCP",
        project: Some("memorypilot"),
        target_id: "mcp-resources-vs-tools",
    },
    FrQuery {
        id: "q-fastembed-batch",
        query: "taille de batch optimale pour fastembed sur CPU",
        project: Some("memorypilot"),
        target_id: "fastembed-batch-size",
    },
    FrQuery {
        id: "q-onnx-providers",
        query: "accélération CoreML pour ONNX Runtime sur Mac M1",
        project: Some("memorypilot"),
        target_id: "onnx-runtime-execution-providers",
    },
    FrQuery {
        id: "q-faiss-usearch",
        query: "alternatives à FAISS pour vector search en Rust",
        project: Some("memorypilot"),
        target_id: "vector-db-faiss-vs-usearch",
    },
    FrQuery {
        id: "q-norm-l2",
        query: "faut-il normaliser les vecteurs avant similarité cosinus",
        project: Some("memorypilot"),
        target_id: "embeddings-normalize-l2",
    },
    FrQuery {
        id: "q-ce-pipeline",
        query: "rerank avec un cross-encoder après BM25 vector",
        project: Some("memorypilot"),
        target_id: "rerank-cross-encoder-late",
    },
    FrQuery {
        id: "q-telemetry-format",
        query: "format de logs structurés pour les recherches",
        project: Some("memorypilot"),
        target_id: "telemetry-jsonl-shipping",
    },
    FrQuery {
        id: "q-mcp-cursor",
        query: "pagination avec curseur opaque dans le protocole MCP",
        project: Some("memorypilot"),
        target_id: "mcp-pagination-cursor",
    },
    FrQuery {
        id: "q-rsc",
        query: "réduire la taille du bundle JavaScript avec React Server Components",
        project: Some("notegenius"),
        target_id: "react-server-components-vs-client",
    },
    FrQuery {
        id: "q-ws-sse",
        query: "WebSocket ou Server Sent Events pour notifications",
        project: Some("notegenius"),
        target_id: "websocket-vs-sse-temps-reel",
    },
    FrQuery {
        id: "q-trpc-graphql",
        query: "API typée bout en bout dans un monorepo TypeScript",
        project: Some("notegenius"),
        target_id: "graphql-vs-trpc-typed-rpc",
    },
    FrQuery {
        id: "q-playwright-cypress",
        query: "tests end-to-end multi-navigateur",
        project: Some("notegenius"),
        target_id: "playwright-vs-cypress-e2e",
    },
    FrQuery {
        id: "q-vitest-jest",
        query: "remplacer Jest par un runner de tests plus rapide",
        project: Some("notegenius"),
        target_id: "vitest-vs-jest-unit",
    },
    FrQuery {
        id: "q-pnpm",
        query: "gestionnaire de paquets monorepo économe en disque",
        project: Some("notegenius"),
        target_id: "yarn-pnpm-vs-npm",
    },
    FrQuery {
        id: "q-design-tokens",
        query: "synchroniser les variables Figma avec le code",
        project: Some("notegenius"),
        target_id: "design-tokens-figma-style-dictionary",
    },
    FrQuery {
        id: "q-aria-live",
        query: "annoncer un message aux lecteurs d'écran",
        project: Some("notegenius"),
        target_id: "accessibilite-aria-live",
    },
    FrQuery {
        id: "q-icu-plural",
        query: "pluralisation correcte dans les traductions françaises",
        project: Some("notegenius"),
        target_id: "i18n-icu-message-format",
    },
    FrQuery {
        id: "q-dark-mode",
        query: "respecter la préférence système pour le thème sombre",
        project: Some("notegenius"),
        target_id: "dark-mode-prefers-color-scheme",
    },
    FrQuery {
        id: "q-csrf",
        query: "protection contre le CSRF pour une SPA moderne",
        project: Some("notegenius"),
        target_id: "csrf-double-submit-cookie",
    },
    FrQuery {
        id: "q-jwt-refresh",
        query: "rotation des refresh tokens et détection de vol",
        project: Some("notegenius"),
        target_id: "jwt-rotation-refresh",
    },
    FrQuery {
        id: "q-passkey",
        query: "remplacer les mots de passe par WebAuthn",
        project: Some("notegenius"),
        target_id: "passkey-webauthn-replace-password",
    },
    FrQuery {
        id: "q-rate-limit",
        query: "limiter le débit des requêtes API avec Redis",
        project: Some("notegenius"),
        target_id: "rate-limit-token-bucket",
    },
    FrQuery {
        id: "q-bullmq",
        query: "système de jobs en arrière plan auto-hébergé",
        project: Some("notegenius"),
        target_id: "queue-bullmq-vs-trigger-dev",
    },
    FrQuery {
        id: "q-monorepo",
        query: "Turborepo ou Nx pour un monorepo TypeScript",
        project: Some("notegenius"),
        target_id: "monorepo-turbo-vs-nx",
    },
    FrQuery {
        id: "q-sentry-maps",
        query: "stack traces lisibles en production avec Sentry",
        project: Some("notegenius"),
        target_id: "sentry-source-maps-upload",
    },
    FrQuery {
        id: "q-pino",
        query: "logger JSON ultra rapide pour Node.js",
        project: Some("notegenius"),
        target_id: "log-structured-pino",
    },
    FrQuery {
        id: "q-tracing-rs",
        query: "instrumenter du code Rust avec des spans",
        project: Some("memorypilot"),
        target_id: "tracing-rust-spans",
    },
    FrQuery {
        id: "q-axum",
        query: "framework web moderne en Rust pour nouvelle API",
        project: Some("memorypilot"),
        target_id: "actix-vs-axum-rust-web",
    },
    FrQuery {
        id: "q-wasm-pack",
        query: "compiler du Rust vers WebAssembly pour le navigateur",
        project: Some("memorypilot"),
        target_id: "wasm-target-rust-wasm-pack",
    },
    FrQuery {
        id: "q-webgpu",
        query: "compute shaders dans le navigateur pour ML",
        project: Some("memorypilot"),
        target_id: "webgpu-compute-vs-webgl",
    },
    FrQuery {
        id: "q-swr-cache",
        query: "servir une réponse cachée pendant le refresh en arrière plan",
        project: Some("notegenius"),
        target_id: "browser-cache-stale-while-revalidate",
    },
    FrQuery {
        id: "q-avif",
        query: "format d'image plus léger que WebP pour le web",
        project: Some("notegenius"),
        target_id: "image-avif-vs-webp",
    },
    FrQuery {
        id: "q-ffmpeg-thumb",
        query: "générer rapidement une vignette depuis une vidéo",
        project: Some("notegenius"),
        target_id: "ffmpeg-thumbnail-fast",
    },
    FrQuery {
        id: "q-im-resize",
        query: "redimensionner un dossier d'images en lot",
        project: Some("notegenius"),
        target_id: "imagemagick-batch-resize",
    },
    FrQuery {
        id: "q-edge-deno",
        query: "fonctions serverless Supabase basées sur Deno",
        project: Some("notegenius"),
        target_id: "supabase-edge-functions-deno",
    },
    FrQuery {
        id: "q-r2",
        query: "stockage objet sans frais de bande passante sortante",
        project: Some("notegenius"),
        target_id: "cloudflare-r2-vs-s3",
    },
    FrQuery {
        id: "q-caa",
        query: "verrouiller quelle autorité peut émettre un certificat TLS",
        project: Some("notegenius"),
        target_id: "dns-caa-record-letsencrypt",
    },
    FrQuery {
        id: "q-spf-dkim",
        query: "configurer la délivrabilité d'un domaine email",
        project: Some("notegenius"),
        target_id: "smtp-spf-dkim-dmarc",
    },
    FrQuery {
        id: "q-resend",
        query: "service d'envoi email transactionnel pour startup",
        project: Some("notegenius"),
        target_id: "postmark-resend-vs-sendgrid",
    },
    FrQuery {
        id: "q-deeplink",
        query: "ouvrir l'app native plutôt que Safari sur un lien",
        project: Some("planify"),
        target_id: "deeplink-universal-links-android-app-links",
    },
    FrQuery {
        id: "q-push-fcm",
        query: "notifications push iOS et Android avec une seule API",
        project: Some("planify"),
        target_id: "push-notifications-fcm-apns",
    },
    FrQuery {
        id: "q-iap-receipt",
        query: "valider un achat in-app côté serveur pour éviter la fraude",
        project: Some("planify"),
        target_id: "in-app-purchase-receipt-validation",
    },
    FrQuery {
        id: "q-rn-flutter",
        query: "choisir entre React Native et Flutter pour MVP mobile",
        project: Some("planify"),
        target_id: "react-native-vs-flutter",
    },
    FrQuery {
        id: "q-clap",
        query: "parser des arguments en ligne de commande Rust avec macros",
        project: Some("memorypilot"),
        target_id: "cli-clap-derive-rust",
    },
    FrQuery {
        id: "q-ratatui",
        query: "construire une interface terminal interactive en Rust",
        project: Some("memorypilot"),
        target_id: "tui-ratatui-rust",
    },
    FrQuery {
        id: "q-watch-loop",
        query: "relancer cargo automatiquement à chaque modification",
        project: Some("memorypilot"),
        target_id: "watchexec-cargo-watch",
    },
    FrQuery {
        id: "q-criterion",
        query: "microbenchmarks Rust avec détection de régression",
        project: Some("memorypilot"),
        target_id: "criterion-bench-rust",
    },
    FrQuery {
        id: "q-musl",
        query: "binaire Rust statique pour image Docker minimale",
        project: Some("memorypilot"),
        target_id: "musl-vs-glibc-static",
    },
    FrQuery {
        id: "q-ssh-jump",
        query: "se connecter à un serveur via un bastion SSH",
        project: Some("memorypilot"),
        target_id: "ssh-config-jump-host",
    },
    FrQuery {
        id: "q-tailscale",
        query: "VPN mesh zero-config pour équipe technique",
        project: Some("memorypilot"),
        target_id: "vpn-wireguard-vs-tailscale",
    },
    FrQuery {
        id: "q-1password",
        query: "gestionnaire de mots de passe partagé avec intégration SSH",
        project: Some("memorypilot"),
        target_id: "1password-vs-bitwarden-team",
    },
    FrQuery {
        id: "q-doppler",
        query: "synchroniser les variables d'environnement secrètes vers Vercel",
        project: Some("memorypilot"),
        target_id: "doppler-vs-aws-secrets-manager",
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
        // Use the scenario.id as the memory id so the benchmark is
        // deterministic across runs. The id is now part of the
        // tie-break in `search`, so a random UUID would reshuffle the
        // bottom of the top-K when scores are very close.
        let pinned_id = format!("fr-bench:{}", spec.id);
        let (memory, _) = db.add_memory_with_id(
            Some(&pinned_id),
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
    // Wait for the ANN index to finish hydrating in RAM. Without this
    // the bench picks up a different code path (full SQL scan vs ANN
    // top-K) on each run depending on the warm-up timing — that was
    // the source of the ±10pp R@5 jitter observed across consecutive
    // runs of `--benchmark-fr`.
    db.wait_for_ann_warm(std::time::Duration::from_secs(60));
    // Pre-hydrate the cross-encoder so the first French query (which
    // adaptive mode reranks) does not pay the ~1.1 GB ONNX init cost
    // synchronously. Without this the very first query of every
    // process took an extra ~5 s, biasing both latency and the
    // ordering of identical-score candidates on subsequent queries.
    crate::reranking::warmup_cross_reranker();

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
            "name": "memorypilot-fr-v2",
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
