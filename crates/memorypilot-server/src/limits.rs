//! Request rate per API key: a token bucket holding up to one minute of
//! requests, refilled continuously. Kept in memory, so each server replica
//! enforces it on its own share of the traffic.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use uuid::Uuid;

pub const DEFAULT_REQUESTS_PER_MINUTE: u32 = 600;
pub const DEFAULT_MAX_MEMORIES: u64 = 100_000;

/// A tenant's limits; `None` falls back to the server's defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TenantLimits {
    pub requests_per_minute: Option<u32>,
    pub max_memories: Option<u64>,
}

struct Bucket {
    tokens: f64,
    updated: Instant,
}

#[derive(Default)]
pub(crate) struct RateLimiter {
    buckets: Mutex<HashMap<Uuid, Bucket>>,
}

impl RateLimiter {
    /// Takes one request from `key`'s bucket: the whole requests still
    /// available, or the seconds to wait before the next one.
    pub(crate) fn take(&self, key: Uuid, per_minute: u32, now: Instant) -> Result<u32, u64> {
        let capacity = f64::from(per_minute.max(1));
        let per_second = capacity / 60.0;
        let Ok(mut buckets) = self.buckets.lock() else {
            return Ok(0);
        };
        let bucket = buckets.entry(key).or_insert(Bucket {
            tokens: capacity,
            updated: now,
        });
        let elapsed = now.saturating_duration_since(bucket.updated).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * per_second).min(capacity);
        bucket.updated = now;
        if bucket.tokens < 1.0 {
            return Err(((1.0 - bucket.tokens) / per_second).ceil().max(1.0) as u64);
        }
        bucket.tokens -= 1.0;
        Ok(bucket.tokens.floor() as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_key_gets_a_minute_of_requests_then_waits_for_the_refill() {
        let limiter = RateLimiter::default();
        let (key, other) = (Uuid::new_v4(), Uuid::new_v4());
        let start = Instant::now();
        assert_eq!(limiter.take(key, 3, start), Ok(2));
        assert_eq!(limiter.take(key, 3, start), Ok(1));
        assert_eq!(limiter.take(key, 3, start), Ok(0));
        assert_eq!(
            limiter.take(key, 3, start),
            Err(20),
            "one request every 20 s"
        );
        assert_eq!(limiter.take(other, 3, start), Ok(2), "buckets are per key");
        assert_eq!(limiter.take(key, 3, start + Duration::from_secs(20)), Ok(0));
        assert_eq!(
            limiter.take(key, 3, start + Duration::from_secs(600)),
            Ok(2),
            "never more than a minute's worth"
        );
    }
}
