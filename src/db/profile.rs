//! Top of the memory pyramid: a cross-project user profile.
//!
//! Levels below already exist: working memory (session), raw memories,
//! hourly / daily / monthly episodes, and the per-project brain. This
//! level answers "who is this user and how do they work", across every
//! project, in a few hundred tokens: durable preferences, active
//! projects, recurring tech, learned skills and recent key decisions.
//! Read-only and deterministic: it is rebuilt from the rows on each call.

use std::collections::HashSet;

use rusqlite::params;
use serde_json::{json, Value};

use super::Database;

fn clip(text: &str, max_chars: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max_chars {
        return flat;
    }
    let clipped: String = flat.chars().take(max_chars).collect();
    format!("{}…", clipped.trim_end())
}

impl Database {
    pub fn get_user_profile(&self, max_tokens: Option<usize>) -> Result<Value, String> {
        let budget = max_tokens.unwrap_or(1200).clamp(200, 6000) * 4; // ~4 chars / token
        let mut used = 0usize;
        let now = chrono::Utc::now();
        let since_30 = (now - chrono::Duration::days(30)).to_rfc3339();
        let since_14 = (now - chrono::Duration::days(14)).to_rfc3339();
        let now_s = now.to_rfc3339();

        // 1. Durable preferences (how the user wants things done).
        let mut preferences = Vec::new();
        {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT content, project FROM memories
                     WHERE kind = 'preference' AND (expires_at IS NULL OR expires_at > ?1)
                     ORDER BY importance DESC, access_count DESC, updated_at DESC LIMIT 120",
                )
                .map_err(|e| format!("profile prefs: {}", e))?;
            let rows = stmt
                .query_map(params![now_s], |r| {
                    Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))
                })
                .map_err(|e| format!("profile prefs: {}", e))?;
            let mut seen: Vec<String> = Vec::new();
            for (content, project) in rows.flatten() {
                let normalized = Self::normalize(&content);
                if seen.iter().any(|s| Self::similarity(s, &normalized) >= 0.6) {
                    continue;
                }
                let line = clip(&content, 220);
                if used + line.len() > budget * 45 / 100 {
                    break;
                }
                used += line.len();
                seen.push(normalized);
                preferences.push(json!({ "text": line, "project": project }));
            }
        }

        // 2. Active projects over the last 30 days.
        let mut projects = Vec::new();
        {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT project, COUNT(*) AS n, MAX(updated_at) FROM memories
                     WHERE project IS NOT NULL AND project != '' AND updated_at >= ?1 AND kind != 'transcript'
                     GROUP BY project ORDER BY n DESC LIMIT 10",
                )
                .map_err(|e| format!("profile projects: {}", e))?;
            let rows = stmt
                .query_map(params![since_30], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                })
                .map_err(|e| format!("profile projects: {}", e))?;
            for (project, count, last) in rows.flatten() {
                used += project.len() + 24;
                projects.push(json!({
                    "project": project,
                    "memories_30d": count,
                    "last_activity": &last[..last.len().min(10)],
                }));
            }
        }

        // 3. Recurring technologies.
        let mut tech = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare(
            "SELECT entity_value, COUNT(DISTINCT memory_id) AS n FROM memory_entities
             WHERE entity_kind = 'tech' GROUP BY entity_value ORDER BY n DESC LIMIT 15",
        ) {
            if let Ok(rows) = stmt.query_map([], |r| r.get::<_, String>(0)) {
                for value in rows.flatten() {
                    used += value.len() + 2;
                    tech.push(value);
                }
            }
        }

        // 4. Learned skills (procedural memory).
        let mut skills = Vec::new();
        if let Ok(mut stmt) = self.conn.prepare(
            "SELECT id, project, metadata, content FROM memories WHERE kind = 'skill'
             ORDER BY updated_at DESC LIMIT 12",
        ) {
            if let Ok(rows) = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, String>(3)?,
                ))
            }) {
                for (id, project, metadata, content) in rows.flatten() {
                    let title = metadata
                        .as_deref()
                        .and_then(|m| serde_json::from_str::<Value>(m).ok())
                        .and_then(|m| m.get("title").and_then(|t| t.as_str()).map(String::from))
                        .unwrap_or_else(|| clip(content.lines().next().unwrap_or(""), 80));
                    if used + title.len() > budget * 85 / 100 {
                        break;
                    }
                    used += title.len() + 40;
                    skills.push(json!({ "id": id, "project": project, "title": title }));
                }
            }
        }

        // 5. Recent key decisions / milestones (last 14 days, importance >= 4).
        let mut recent = Vec::new();
        {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT content, project, kind, updated_at FROM memories
                     WHERE kind IN ('decision','milestone','architecture') AND importance >= 4
                       AND updated_at >= ?1
                     ORDER BY updated_at DESC LIMIT 40",
                )
                .map_err(|e| format!("profile recent: {}", e))?;
            let rows = stmt
                .query_map(params![since_14], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, Option<String>>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                    ))
                })
                .map_err(|e| format!("profile recent: {}", e))?;
            let mut seen_projects: HashSet<Option<String>> = HashSet::new();
            for (content, project, kind, updated) in rows.flatten() {
                // One line per project keeps the overview broad.
                if !seen_projects.insert(project.clone()) {
                    continue;
                }
                let line = clip(&content, 180);
                if used + line.len() > budget {
                    break;
                }
                used += line.len();
                recent.push(json!({
                    "project": project,
                    "kind": kind,
                    "date": &updated[..updated.len().min(10)],
                    "text": line,
                }));
                if recent.len() >= 8 {
                    break;
                }
            }
        }

        Ok(json!({
            "level": "user_profile",
            "preferences": preferences,
            "active_projects": projects,
            "tech": tech,
            "skills": skills,
            "recent_key_changes": recent,
            "approx_tokens": used / 4,
        }))
    }
}
