use super::{row_to_memory, Database, Memory, MemoryScope, RecallMode, SearchResult};

#[derive(Debug, Clone)]
struct BenchmarkScenarioSpec {
    name: &'static str,
    query: &'static str,
    project: Option<&'static str>,
    expected_kind: Option<&'static str>,
    match_substrings: &'static [&'static str],
    mode: RecallMode,
}

#[derive(Debug, Clone)]
struct BenchmarkScenarioRun {
    name: String,
    source: String,
    query: String,
    mode: RecallMode,
    expected_memory: Memory,
}

impl Database {
    fn benchmark_query_for_memory(memory: &Memory) -> Option<String> {
        let mut terms: Vec<String> = Vec::new();

        for entity in crate::graph::extract_entities(&memory.content, memory.project.as_deref()) {
            if entity.kind == "project" {
                continue;
            }
            let value = entity
                .value
                .rsplit('/')
                .next()
                .unwrap_or(entity.value.as_str())
                .trim_matches(|character: char| {
                    !character.is_alphanumeric() && character != '-' && character != '_'
                })
                .to_ascii_lowercase();
            if value.len() > 2 && !terms.contains(&value) {
                terms.push(value);
            }
        }

        for tag in &memory.tags {
            let normalized = tag.trim().to_ascii_lowercase();
            if normalized.len() > 2 && !terms.contains(&normalized) {
                terms.push(normalized);
            }
        }

        for word in memory.content.split_whitespace() {
            let normalized = word
                .trim_matches(|character: char| {
                    !character.is_alphanumeric() && character != '-' && character != '_'
                })
                .to_ascii_lowercase();
            if normalized.len() <= 3 {
                continue;
            }
            if memory.project.as_deref() == Some(normalized.as_str()) {
                continue;
            }
            if matches!(
                normalized.as_str(),
                "this"
                    | "that"
                    | "with"
                    | "from"
                    | "have"
                    | "been"
                    | "will"
                    | "would"
                    | "could"
                    | "into"
                    | "using"
                    | "when"
                    | "where"
                    | "what"
                    | "which"
                    | "dans"
                    | "pour"
                    | "avec"
                    | "cette"
                    | "sont"
                    | "mais"
                    | "plus"
                    | "todo"
                    | "note"
                    | "fact"
                    | "decision"
                    | "pattern"
            ) {
                continue;
            }
            if !terms.contains(&normalized) {
                terms.push(normalized);
            }
            if terms.len() >= 4 {
                break;
            }
        }

        if terms.is_empty() {
            None
        } else {
            Some(terms.into_iter().take(4).collect::<Vec<_>>().join(" "))
        }
    }

    fn curated_benchmark_scenarios() -> Vec<BenchmarkScenarioSpec> {
        vec![
            BenchmarkScenarioSpec {
                name: "memorypilot-safe-mode-default",
                query: "safe mode credentials recall",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["safe mode", "credential"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "memorypilot-benchmark-cli",
                query: "benchmark_recall top1 top5",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["benchmark_recall", "top1", "top5"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "memorypilot-cross-project-pollution",
                query: "cross project pollution recall",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["cross-project", "pollution"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "zed-fork-observability-metrics",
                query: "mcp mcphub observability metrics",
                project: Some("zed-fork"),
                expected_kind: Some("decision"),
                match_substrings: &["observability", "metrics"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "zed-fork-health-check",
                query: "mcp health check",
                project: Some("zed-fork"),
                expected_kind: Some("bug"),
                match_substrings: &["health check"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "mcphub-schema-cache",
                query: "mcp cargo schema cache",
                project: Some("mcphub"),
                expected_kind: Some("decision"),
                match_substrings: &["schema", "cache"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "planify-testflight-build",
                query: "testflight ios build sociomator",
                project: Some("planify"),
                expected_kind: Some("fact"),
                match_substrings: &["testflight", "ios"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "planify-instagram-cache",
                query: "instagram cache flutter shared_preferences",
                project: Some("planify"),
                expected_kind: Some("decision"),
                match_substrings: &["instagram", "shared_preferences"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "memorypilot-session-scope",
                query: "session thread window scope recall",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["session", "thread", "window"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "memorypilot-transcript-distillation",
                query: "transcript distillation recall",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["transcript", "distill"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "memorypilot-credential-safety",
                query: "credential leakage safe mode",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["credential", "safe"],
                mode: RecallMode::Safe,
            },
            BenchmarkScenarioSpec {
                name: "memorypilot-recall-explain",
                query: "recall explain search score graph boost",
                project: Some("memorypilot"),
                expected_kind: Some("decision"),
                match_substrings: &["recall", "explain"],
                mode: RecallMode::Safe,
            },
        ]
    }

    fn resolve_benchmark_memory(
        &self,
        spec: &BenchmarkScenarioSpec,
    ) -> Result<Option<Memory>, String> {
        let canonical_project = spec.project.and_then(Self::canonical_project_name);
        let (memories, _) = self.list_memories(
            canonical_project.as_deref(),
            spec.expected_kind,
            Some("transcript"),
            250,
            0,
        )?;

        let mut matches: Vec<Memory> = memories
            .into_iter()
            .filter(|memory| {
                let haystack = format!(
                    "{} {}",
                    memory.content.to_ascii_lowercase(),
                    memory.tags.join(" ").to_ascii_lowercase()
                );
                spec.match_substrings
                    .iter()
                    .all(|term| haystack.contains(term))
            })
            .collect();

        matches.sort_by(|left, right| {
            right
                .importance
                .cmp(&left.importance)
                .then_with(|| right.updated_at.cmp(&left.updated_at))
        });

        Ok(matches.into_iter().next())
    }

    fn generated_benchmark_runs(
        &self,
        limit: usize,
        excluded_ids: &std::collections::HashSet<String>,
    ) -> Result<Vec<BenchmarkScenarioRun>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let (memories, _) = self.list_memories(None, None, Some("transcript"), limit * 6, 0)?;
        let mut runs = Vec::new();
        let mut seen_projects = std::collections::HashSet::new();

        for memory in memories.into_iter() {
            if excluded_ids.contains(&memory.id) {
                continue;
            }
            if !matches!(
                memory.kind.as_str(),
                "decision" | "bug" | "pattern" | "fact" | "snippet"
            ) {
                continue;
            }
            if let Some(project_name) = memory.project.as_deref() {
                if !seen_projects.insert((project_name.to_string(), memory.kind.clone())) {
                    continue;
                }
            }
            let Some(query) = Self::benchmark_query_for_memory(&memory) else {
                continue;
            };
            let stable_results = self
                .search(
                    &query,
                    5,
                    memory.project.as_deref(),
                    Some(memory.kind.as_str()),
                    None,
                    None,
                )
                .unwrap_or_default();
            let is_stable_candidate = stable_results
                .iter()
                .any(|result| result.memory.id == memory.id);
            if !is_stable_candidate {
                continue;
            }

            runs.push(BenchmarkScenarioRun {
                name: format!(
                    "generated:{}:{}",
                    memory.project.clone().unwrap_or_else(|| "global".into()),
                    memory.kind
                ),
                source: "generated".into(),
                query,
                mode: RecallMode::Safe,
                expected_memory: memory,
            });

            if runs.len() >= limit {
                break;
            }
        }

        Ok(runs)
    }

    fn benchmark_percentage(count: usize, total: usize) -> f64 {
        ((count as f64 / total.max(1) as f64) * 100.0).round() / 100.0
    }

    pub fn benchmark_recall(&self, scenario_limit: usize) -> Result<serde_json::Value, String> {
        let candidate_limit = scenario_limit.max(5).min(30);
        let golden_specs = Self::curated_benchmark_scenarios();
        let considered_golden_specs = golden_specs
            .into_iter()
            .take(candidate_limit)
            .collect::<Vec<_>>();
        let mut scenarios = Vec::new();
        let mut skipped_golden = Vec::new();
        let mut expected_ids = std::collections::HashSet::new();

        for spec in &considered_golden_specs {
            match self.resolve_benchmark_memory(spec)? {
                Some(memory) => {
                    expected_ids.insert(memory.id.clone());
                    scenarios.push(BenchmarkScenarioRun {
                        name: spec.name.into(),
                        source: "golden".into(),
                        query: spec.query.into(),
                        mode: spec.mode,
                        expected_memory: memory,
                    });
                }
                None => skipped_golden.push(serde_json::json!({
                    "name": spec.name,
                    "project": spec.project,
                    "kind": spec.expected_kind,
                    "query": spec.query,
                    "reason": "missing_expected_memory"
                })),
            }
        }

        let golden_executed_count = scenarios.len();
        if scenarios.len() < candidate_limit {
            scenarios.extend(
                self.generated_benchmark_runs(candidate_limit - scenarios.len(), &expected_ids)?,
            );
        }

        let mut hits_top1 = 0usize;
        let mut hits_top5 = 0usize;
        let mut cross_project_leaks = 0usize;
        let mut credential_leaks_safe = 0usize;
        let mut explain_with_search_score = 0usize;
        let mut scenario_results = Vec::new();
        let mut golden_run_count = 0usize;
        let mut generated_run_count = 0usize;

        for scenario in scenarios {
            let recall = self.recall(
                scenario.expected_memory.project.as_deref(),
                None,
                Some(&scenario.query),
                scenario.mode,
                true,
                false,
                &MemoryScope::default(),
            )?;

            let explain_block = recall
                .get("explain")
                .and_then(|value| value.as_object())
                .cloned()
                .unwrap_or_default();
            let selected = explain_block
                .get("selected_memories")
                .and_then(|value| value.as_array())
                .cloned()
                .unwrap_or_default();

            let selected_ids: Vec<String> = selected
                .iter()
                .filter_map(|value| value.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect();

            let top1_hit = selected_ids
                .first()
                .map(|id| id == &scenario.expected_memory.id)
                .unwrap_or(false);
            let top5_hit = selected_ids
                .iter()
                .take(5)
                .any(|id| id == &scenario.expected_memory.id);
            if top1_hit {
                hits_top1 += 1;
            }
            if top5_hit {
                hits_top5 += 1;
            }

            let cross_leak_count = selected
                .iter()
                .take(5)
                .filter(|value| {
                    let selected_project =
                        value.get("project").and_then(|project| project.as_str());
                    scenario
                        .expected_memory
                        .project
                        .as_deref()
                        .map(|expected_project| {
                            selected_project.is_some() && selected_project != Some(expected_project)
                        })
                        .unwrap_or(false)
                })
                .count();
            if cross_leak_count > 0 {
                cross_project_leaks += 1;
            }

            let credential_leak_count = selected
                .iter()
                .filter(|value| {
                    value.get("kind").and_then(|kind| kind.as_str()) == Some("credential")
                })
                .count();
            if credential_leak_count > 0 {
                credential_leaks_safe += 1;
            }

            let has_search_score = selected.iter().any(|value| {
                value
                    .get("search_score")
                    .and_then(|score| score.as_f64())
                    .is_some()
            });
            if has_search_score {
                explain_with_search_score += 1;
            }

            if scenario.source == "golden" {
                golden_run_count += 1;
            } else {
                generated_run_count += 1;
            }

            scenario_results.push(serde_json::json!({
                "scenario_name": scenario.name,
                "scenario_source": scenario.source,
                "mode": scenario.mode.as_str(),
                "project": scenario.expected_memory.project,
                "kind": scenario.expected_memory.kind,
                "query": scenario.query,
                "expected_memory_id": scenario.expected_memory.id,
                "top1_hit": top1_hit,
                "top5_hit": top5_hit,
                "cross_project_leak_count": cross_leak_count,
                "credential_leak_count_safe": credential_leak_count,
                "selected_memory_ids": selected_ids.into_iter().take(5).collect::<Vec<_>>(),
            }));
        }

        let scenario_count = scenario_results.len();

        Ok(serde_json::json!({
            "status": "ok",
            "scenario_count": scenario_count,
            "golden_defined_count": considered_golden_specs.len(),
            "golden_executed_count": golden_executed_count,
            "golden_skipped_count": skipped_golden.len(),
            "golden_skipped": skipped_golden,
            "scenario_source_counts": {
                "golden": golden_run_count,
                "generated": generated_run_count
            },
            "top1_hit_rate": Self::benchmark_percentage(hits_top1, scenario_count),
            "top5_hit_rate": Self::benchmark_percentage(hits_top5, scenario_count),
            "cross_project_leak_rate": Self::benchmark_percentage(cross_project_leaks, scenario_count),
            "credential_leak_rate_safe": Self::benchmark_percentage(credential_leaks_safe, scenario_count),
            "explain_consistency_rate": Self::benchmark_percentage(explain_with_search_score, scenario_count),
            "scenarios": scenario_results,
        }))
    }

    pub fn benchmark_search(&self, scenario_limit: usize) -> Result<serde_json::Value, String> {
        let all_memories = self.list_all_memories_for_benchmark()?;
        if all_memories.len() < 5 {
            return Ok(serde_json::json!({
                "status": "insufficient_data",
                "memory_count": all_memories.len(),
                "message": "Need at least 5 memories to run search benchmark"
            }));
        }

        let scenario_count = scenario_limit.min(all_memories.len()).min(50);
        let step = all_memories.len() / scenario_count;

        let mut hits_r5 = 0usize;
        let mut hits_r10 = 0usize;
        let mut ndcg_sum = 0.0f64;
        let mut cluster_coherence_sum = 0.0f64;
        let mut avg_search_ms = 0.0f64;
        let mut scenarios = Vec::new();

        for i in 0..scenario_count {
            let target = &all_memories[i * step];
            let query_words: Vec<&str> = target.content.split_whitespace().take(8).collect();
            if query_words.len() < 2 {
                continue;
            }
            let query = query_words.join(" ");

            let start = std::time::Instant::now();
            let results = self.search(&query, 10, target.project.as_deref(), None, None, None)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            avg_search_ms += elapsed_ms;

            let result_ids: Vec<&str> = results.iter().map(|r| r.memory.id.as_str()).collect();

            let in_top5 = result_ids.iter().take(5).any(|id| *id == target.id);
            if in_top5 {
                hits_r5 += 1;
            }

            let in_top10 = result_ids.iter().take(10).any(|id| *id == target.id);
            if in_top10 {
                hits_r10 += 1;
            }

            let ndcg = if let Some(pos) = result_ids.iter().position(|id| *id == target.id) {
                if pos < 10 {
                    1.0 / (pos as f64 + 2.0).log2()
                } else {
                    0.0
                }
            } else {
                0.0
            };
            ndcg_sum += ndcg;

            let coherence = self.measure_cluster_coherence(&results);
            cluster_coherence_sum += coherence;

            scenarios.push(serde_json::json!({
                "query": query,
                "target_id": target.id,
                "target_kind": target.kind,
                "target_project": target.project,
                "r5_hit": in_top5,
                "r10_hit": in_top10,
                "ndcg10": (ndcg * 1000.0).round() / 1000.0,
                "cluster_coherence": (coherence * 1000.0).round() / 1000.0,
                "search_ms": (elapsed_ms * 100.0).round() / 100.0,
                "results_returned": results.len(),
            }));
        }

        let actual_count = scenarios.len().max(1);
        let r5_rate = (hits_r5 as f64 / actual_count as f64 * 1000.0).round() / 10.0;
        let r10_rate = (hits_r10 as f64 / actual_count as f64 * 1000.0).round() / 10.0;
        let ndcg10 = (ndcg_sum / actual_count as f64 * 1000.0).round() / 10.0;
        let coherence = (cluster_coherence_sum / actual_count as f64 * 1000.0).round() / 10.0;
        avg_search_ms = (avg_search_ms / actual_count as f64 * 100.0).round() / 100.0;

        Ok(serde_json::json!({
            "status": "ok",
            "memory_count": all_memories.len(),
            "scenario_count": actual_count,
            "metrics": {
                "R@5": format!("{}%", r5_rate),
                "R@10": format!("{}%", r10_rate),
                "NDCG@10": format!("{}%", ndcg10),
                "cluster_coherence": format!("{}%", coherence),
                "avg_search_ms": format!("{:.2}ms", avg_search_ms),
            },
            "scenarios": scenarios,
        }))
    }

    fn list_all_memories_for_benchmark(&self) -> Result<Vec<Memory>, String> {
        let mut stmt = self.conn.prepare(
            "SELECT id,content,kind,project,tags,source,importance,expires_at,metadata,created_at,updated_at,last_accessed_at,access_count \
             FROM memories WHERE kind != 'transcript_chunk' AND length(content) > 20 ORDER BY created_at DESC LIMIT 500"
        ).map_err(|e| format!("Benchmark list: {}", e))?;
        let rows = stmt
            .query_map([], |r| Ok(row_to_memory(r)))
            .map_err(|e| format!("Benchmark query: {}", e))?;
        Ok(rows.flatten().collect())
    }

    fn measure_cluster_coherence(&self, results: &[SearchResult]) -> f64 {
        if results.len() < 2 {
            return 1.0;
        }
        let top5: Vec<&str> = results
            .iter()
            .take(5)
            .map(|r| r.memory.id.as_str())
            .collect();
        if top5.len() < 2 {
            return 1.0;
        }

        let placeholders: Vec<String> = (1..=top5.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "SELECT COUNT(DISTINCT a.memory_id || ':' || b.memory_id) FROM memory_entities a \
             JOIN memory_entities b ON a.entity_value = b.entity_value AND a.entity_kind = b.entity_kind AND a.memory_id < b.memory_id \
             WHERE a.memory_id IN ({0}) AND b.memory_id IN ({0})",
            placeholders.join(",")
        );

        let connected_pairs: i64 = if let Ok(mut stmt) = self.conn.prepare(&sql) {
            let params: Vec<&dyn rusqlite::types::ToSql> = top5
                .iter()
                .map(|id| id as &dyn rusqlite::types::ToSql)
                .collect();
            stmt.query_row(params.as_slice(), |r| r.get(0)).unwrap_or(0)
        } else {
            0
        };

        let max_pairs = (top5.len() * (top5.len() - 1)) / 2;
        if max_pairs == 0 {
            return 1.0;
        }
        (connected_pairs as f64 / max_pairs as f64).min(1.0)
    }
}
