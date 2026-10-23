# Phase 8 : benchmarks

## Dans le cœur (local, SQLite)

- Français : `scripts/bench-gate.sh` (rappel@10, MRR).
- LongMemEval-S : `MemoryPilot --benchmark-longmemeval [PATH] [--limit N]` (`benchmark_longmemeval.rs`). Mesure le retrieval (R@5, R@10, NDCG, MRR) et, avec le lecteur extractif local, un score de réponse. Aucun LLM, aucune clé.
- Pas de génération de réponse par un LLM : ce runner n'existe pas encore. L'écrire demande de choisir un modèle et un budget (décision d'Antoine).
- LoCoMo : aucun runner dans le dépôt.

## Serveur (Postgres)

`scripts/bench-server.sh` (après `scripts/test-postgres.sh`) :

- isolation : un autre tenant ne voit pas les mémoires ;
- péremption : une mémoire expirée ne ressort pas en recherche ;
- latence : 20 recherches, moyenne < 250 ms, sans embedder.

Les tests d'intégration (`tests/postgres.rs`, `tests/api.rs`) couvrent l'isolation entre tenants et utilisateurs finaux (clés d'API et jetons OAuth), plus le benchmark français rejoué sur Postgres.
