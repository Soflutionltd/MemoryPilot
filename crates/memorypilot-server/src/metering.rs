//! Usage counters kept in memory and added to `usage_daily` in batches.

use std::collections::HashMap;
use std::sync::Mutex;

use chrono::NaiveDate;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DailyUsage {
    pub tenant: Uuid,
    pub space: Uuid,
    pub day: NaiveDate,
    pub requests: u64,
    pub writes: u64,
    pub searches: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Usage {
    Request,
    Write,
    Search,
}

#[derive(Default, Clone, Copy)]
struct Counts {
    requests: u64,
    writes: u64,
    searches: u64,
}

#[derive(Default)]
pub(crate) struct Meter {
    counts: Mutex<HashMap<(Uuid, Uuid, NaiveDate), Counts>>,
}

impl Meter {
    pub(crate) fn count(&self, tenant: Uuid, space: Uuid, day: NaiveDate, usage: Usage) {
        let Ok(mut counts) = self.counts.lock() else {
            return;
        };
        let entry = counts.entry((tenant, space, day)).or_default();
        match usage {
            Usage::Request => entry.requests += 1,
            Usage::Write => entry.writes += 1,
            Usage::Search => entry.searches += 1,
        }
    }

    /// Everything counted since the last drain.
    pub(crate) fn drain(&self) -> Vec<DailyUsage> {
        let Ok(mut counts) = self.counts.lock() else {
            return Vec::new();
        };
        counts
            .drain()
            .map(|((tenant, space, day), counts)| DailyUsage {
                tenant,
                space,
                day,
                requests: counts.requests,
                writes: counts.writes,
                searches: counts.searches,
            })
            .collect()
    }

    /// Puts back counts that could not be stored, to try again later.
    pub(crate) fn restore(&self, rows: Vec<DailyUsage>) {
        let Ok(mut counts) = self.counts.lock() else {
            return;
        };
        for row in rows {
            let entry = counts.entry((row.tenant, row.space, row.day)).or_default();
            entry.requests += row.requests;
            entry.writes += row.writes;
            entry.searches += row.searches;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_drain_once_and_can_be_put_back() {
        let meter = Meter::default();
        let (tenant, space) = (Uuid::new_v4(), Uuid::new_v4());
        let day = NaiveDate::from_ymd_opt(2026, 10, 23).expect("date");
        for usage in [Usage::Request, Usage::Request, Usage::Write, Usage::Search] {
            meter.count(tenant, space, day, usage);
        }
        let drained = meter.drain();
        assert_eq!(
            drained,
            vec![DailyUsage {
                tenant,
                space,
                day,
                requests: 2,
                writes: 1,
                searches: 1
            }]
        );
        assert!(meter.drain().is_empty(), "drained once");
        meter.restore(drained);
        meter.count(tenant, space, day, Usage::Request);
        assert_eq!(meter.drain()[0].requests, 3);
    }
}
