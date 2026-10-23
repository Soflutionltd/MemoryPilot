# Phase 8 — benchmarks

## Déjà dans le cœur (local, SQLite)

- Français : `scripts/bench-gate.sh` (rappel@10, MRR).
- LongMemEval-S : dans `memorypilot-core` (`benchmark_longmemeval.rs`). Une clé LLM n'est pas requise pour le score de retrieval (vecteurs + BM25). Le score de génération de bout en bout attend `MEMORYPILOT_LLM_URL` / le budget d'Antoine : ne pas le lancer sans son accord.
- LoCoMo : même règle. Le runner n'est pas lancé ici.

## Serveur (Postgres)

`scripts/bench-server.sh` (après `scripts/test-postgres.sh`) :

- isolation : un autre tenant ne voit pas les mémoires ;
- péremption : une mémoire expirée n'est pas renvoyée ;
- latence : 20 recherches, moyenne < 250 ms à froid sans embedder.

Les tests d'intégration (`tests/postgres.rs`, `tests/api.rs`) couvrent déjà l'isolation entre tenants et utilisateurs finaux, plus le benchmark français rejoué sur Postgres.
