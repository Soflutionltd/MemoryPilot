//! Rule-based temporal grounding — no model involved.
//!
//! Two halves:
//!
//! * [`extract_dates`] finds explicit calendar dates in a memory's text
//!   (ISO, slashed, `12 septembre 2026`, `September 12, 2026`, plus a few
//!   day-relative words resolved against the memory's own timestamp). The
//!   write path stores them as `date` entities so a query can match on
//!   *when* something happened instead of hoping the embedding carries it.
//! * [`parse_query_window`] turns the temporal phrase of a query — `last
//!   week`, `il y a deux semaines`, `en mars`, `last Tuesday`, `2026-09-01`
//!   — into a `[start, end]` day window relative to a reference date. The
//!   ranker boosts candidates dated inside the window and gently demotes
//!   dated candidates that fall clearly outside it.
//!
//! Both are deliberately conservative: a false window costs recall, a
//! missed one only costs the boost.

use chrono::{Datelike, Duration, NaiveDate, Weekday};

/// `MEMORYPILOT_TEMPORAL=0` disables the query window (ablations).
pub fn is_enabled() -> bool {
    !matches!(
        std::env::var("MEMORYPILOT_TEMPORAL").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DateWindow {
    pub start: NaiveDate,
    pub end: NaiveDate,
}

impl DateWindow {
    fn day(date: NaiveDate) -> Self {
        Self { start: date, end: date }
    }

    fn around(date: NaiveDate, slack_days: i64) -> Self {
        Self {
            start: date - Duration::days(slack_days),
            end: date + Duration::days(slack_days),
        }
    }

    pub fn contains(&self, date: NaiveDate) -> bool {
        date >= self.start && date <= self.end
    }

    pub fn len_days(&self) -> i64 {
        (self.end - self.start).num_days().max(0) + 1
    }

    /// Distance in days from `date` to the nearest edge; 0 when inside.
    pub fn distance_days(&self, date: NaiveDate) -> i64 {
        if date < self.start {
            (self.start - date).num_days()
        } else if date > self.end {
            (date - self.end).num_days()
        } else {
            0
        }
    }
}

const MONTHS: &[(&str, u32)] = &[
    ("january", 1), ("jan", 1), ("janvier", 1), ("janv", 1),
    ("february", 2), ("feb", 2), ("février", 2), ("fevrier", 2), ("fév", 2), ("fev", 2),
    ("march", 3), ("mar", 3), ("mars", 3),
    ("april", 4), ("apr", 4), ("avril", 4), ("avr", 4),
    ("may", 5), ("mai", 5),
    ("june", 6), ("jun", 6), ("juin", 6),
    ("july", 7), ("jul", 7), ("juillet", 7), ("juil", 7),
    ("august", 8), ("aug", 8), ("août", 8), ("aout", 8),
    ("september", 9), ("sept", 9), ("sep", 9), ("septembre", 9),
    ("october", 10), ("oct", 10), ("octobre", 10),
    ("november", 11), ("nov", 11), ("novembre", 11),
    ("december", 12), ("dec", 12), ("décembre", 12), ("decembre", 12), ("déc", 12),
];

const WEEKDAYS: &[(&str, Weekday)] = &[
    ("monday", Weekday::Mon), ("mon", Weekday::Mon), ("lundi", Weekday::Mon),
    ("tuesday", Weekday::Tue), ("tue", Weekday::Tue), ("tues", Weekday::Tue), ("mardi", Weekday::Tue),
    ("wednesday", Weekday::Wed), ("wed", Weekday::Wed), ("mercredi", Weekday::Wed),
    ("thursday", Weekday::Thu), ("thu", Weekday::Thu), ("thurs", Weekday::Thu), ("jeudi", Weekday::Thu),
    ("friday", Weekday::Fri), ("fri", Weekday::Fri), ("vendredi", Weekday::Fri),
    ("saturday", Weekday::Sat), ("sat", Weekday::Sat), ("samedi", Weekday::Sat),
    ("sunday", Weekday::Sun), ("sun", Weekday::Sun), ("dimanche", Weekday::Sun),
];

const NUMBER_WORDS: &[(&str, i64)] = &[
    ("a", 1), ("an", 1), ("one", 1), ("un", 1), ("une", 1),
    ("two", 2), ("deux", 2), ("couple of", 2), ("couple", 2),
    ("three", 3), ("trois", 3), ("few", 3), ("quelques", 3),
    ("four", 4), ("quatre", 4), ("five", 5), ("cinq", 5), ("six", 6),
    ("seven", 7), ("sept", 7), ("eight", 8), ("huit", 8), ("nine", 9), ("neuf", 9),
    ("ten", 10), ("dix", 10), ("eleven", 11), ("onze", 11), ("twelve", 12), ("douze", 12),
];

fn month_number(token: &str) -> Option<u32> {
    let token = token.trim_end_matches('.');
    MONTHS
        .iter()
        .find(|(name, _)| *name == token)
        .map(|(_, number)| *number)
}

fn weekday_of(token: &str) -> Option<Weekday> {
    WEEKDAYS
        .iter()
        .find(|(name, _)| *name == token)
        .map(|(_, weekday)| *weekday)
}

fn number_of(token: &str) -> Option<i64> {
    if let Ok(value) = token.parse::<i64>() {
        return (0..=10_000).contains(&value).then_some(value);
    }
    NUMBER_WORDS
        .iter()
        .find(|(name, _)| *name == token)
        .map(|(_, value)| *value)
}

/// Lower-case word tokens with digits kept, separators dropped. Slashes
/// and dashes inside a date survive as part of one token so `2026-09-16`
/// stays whole; `12/09/2026` too.
fn tokens(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    lower
        .split(|character: char| {
            !(character.is_alphanumeric() || matches!(character, '-' | '/' | '\'' | '.'))
        })
        .map(|token| token.trim_matches(|character: char| matches!(character, '.' | '\'' | '-' | '/')))
        .filter(|token| !token.is_empty())
        .map(String::from)
        .collect()
}

fn parse_numeric_date(token: &str) -> Option<NaiveDate> {
    let mut parts: Vec<&str> = token.split(['-', '/']).collect();
    if parts.len() != 3 {
        return None;
    }
    // `2026-09-16t21` — the day part of a lower-cased RFC 3339 stamp.
    if let Some((day, _)) = parts[2].split_once('t') {
        parts[2] = day;
    }
    if parts.iter().any(|part| part.is_empty() || part.len() > 4) {
        return None;
    }
    let numbers: Option<Vec<u32>> = parts.iter().map(|part| part.parse::<u32>().ok()).collect();
    let numbers = numbers?;
    let (a, b, c) = (numbers[0], numbers[1], numbers[2]);
    // YYYY-MM-DD / YYYY/MM/DD
    if parts[0].len() == 4 {
        return NaiveDate::from_ymd_opt(a as i32, b, c);
    }
    if parts[2].len() != 4 {
        return None;
    }
    let year = c as i32;
    // DD/MM/YYYY unless the first field cannot be a day-of-month.
    if a > 12 {
        NaiveDate::from_ymd_opt(year, b, a)
    } else if b > 12 {
        NaiveDate::from_ymd_opt(year, a, b)
    } else {
        NaiveDate::from_ymd_opt(year, b, a)
    }
}

fn parse_year(token: &str) -> Option<i32> {
    if token.len() == 4 {
        let year = token.parse::<i32>().ok()?;
        return (1970..=2100).contains(&year).then_some(year);
    }
    None
}

fn parse_day(token: &str) -> Option<u32> {
    let digits: String = token.chars().take_while(|character| character.is_ascii_digit()).collect();
    if digits.is_empty() || digits.len() > 2 {
        return None;
    }
    let rest = &token[digits.len()..];
    if !(rest.is_empty() || matches!(rest, "st" | "nd" | "rd" | "th" | "er" | "e")) {
        return None;
    }
    let day = digits.parse::<u32>().ok()?;
    (1..=31).contains(&day).then_some(day)
}

/// Explicit calendar dates mentioned in `text`, deduplicated, in order of
/// appearance. `reference` (the memory's own date) resolves `yesterday`,
/// `today`, `tomorrow` and supplies the year of `September 12`. Without it
/// only fully specified dates are returned.
pub fn extract_dates(text: &str, reference: Option<NaiveDate>) -> Vec<NaiveDate> {
    let tokens = tokens(text);
    let mut found: Vec<NaiveDate> = Vec::new();
    let push = |date: NaiveDate, found: &mut Vec<NaiveDate>| {
        if !found.contains(&date) {
            found.push(date);
        }
    };

    let mut index = 0;
    while index < tokens.len() {
        let token = tokens[index].as_str();

        if let Some(date) = parse_numeric_date(token) {
            push(date, &mut found);
            index += 1;
            continue;
        }

        // `12 septembre 2026` / `12 septembre` / `12th of September`
        if let Some(day) = parse_day(token) {
            let mut cursor = index + 1;
            if cursor < tokens.len() && matches!(tokens[cursor].as_str(), "of" | "de" | "du") {
                cursor += 1;
            }
            if let Some(month) = tokens.get(cursor).and_then(|t| month_number(t)) {
                let year = tokens.get(cursor + 1).and_then(|t| parse_year(t));
                let resolved_year = year.or_else(|| reference.map(|r| r.year()));
                if let Some(date) =
                    resolved_year.and_then(|y| NaiveDate::from_ymd_opt(y, month, day))
                {
                    push(date, &mut found);
                    index = cursor + if year.is_some() { 2 } else { 1 };
                    continue;
                }
            }
        }

        // `September 12, 2026` / `September 12` / `Sept. 2026`
        if let Some(month) = month_number(token) {
            if let Some(day) = tokens.get(index + 1).and_then(|t| parse_day(t)) {
                let year = tokens.get(index + 2).and_then(|t| parse_year(t));
                let resolved_year = year.or_else(|| reference.map(|r| r.year()));
                if let Some(date) =
                    resolved_year.and_then(|y| NaiveDate::from_ymd_opt(y, month, day))
                {
                    push(date, &mut found);
                    index += if year.is_some() { 3 } else { 2 };
                    continue;
                }
            }
        }

        if let Some(reference) = reference {
            let relative = match token {
                "today" | "aujourd'hui" | "aujourd" => Some(0),
                "yesterday" | "hier" => Some(-1),
                "tomorrow" | "demain" => Some(1),
                _ => None,
            };
            if let Some(offset) = relative {
                push(reference + Duration::days(offset), &mut found);
            }
        }

        index += 1;
    }
    found
}

/// The day window a query is asking about, if it carries a temporal
/// phrase. Explicit dates win over relative phrases; the first match in
/// reading order is used when several relative phrases are present.
pub fn parse_query_window(query: &str, today: NaiveDate) -> Option<DateWindow> {
    let tokens = tokens(query);
    if tokens.is_empty() {
        return None;
    }

    // 1. Explicit dates in the query.
    let explicit = extract_dates(query, Some(today));
    if let Some(first) = explicit.first() {
        return Some(DateWindow::around(*first, 1));
    }

    let has = |index: usize, word: &str| tokens.get(index).map(|t| t == word).unwrap_or(false);

    for index in 0..tokens.len() {
        let token = tokens[index].as_str();

        // today / yesterday / tonight …
        match token {
            "today" | "tonight" | "aujourd'hui" | "aujourd" => {
                return Some(DateWindow::day(today));
            }
            "yesterday" | "hier" => {
                return Some(DateWindow::around(today - Duration::days(1), 0));
            }
            _ => {}
        }

        // `N days/weeks/months/years ago` — `il y a N jours/semaines/mois`
        if let Some(count) = number_of(token) {
            if let Some(unit) = tokens.get(index + 1) {
                let unit_is_ago = has(index + 2, "ago") || has(index + 2, "earlier")
                    || (index >= 3 && has(index - 3, "il") && has(index - 2, "y") && has(index - 1, "a"));
                if unit_is_ago {
                    if let Some(window) = window_ago(today, count, unit) {
                        return Some(window);
                    }
                }
            }
        }
        // `a couple of weeks ago`
        if token == "couple" && has(index + 1, "of") {
            if let Some(unit) = tokens.get(index + 2) {
                if has(index + 3, "ago") {
                    if let Some(window) = window_ago(today, 2, unit) {
                        return Some(window);
                    }
                }
            }
        }

        // last / this / past / previous + unit or weekday
        if matches!(token, "last" | "past" | "previous" | "this" | "earlier") {
            if let Some(next) = tokens.get(index + 1) {
                let is_this = token == "this";
                match next.as_str() {
                    "week" => {
                        return Some(if is_this {
                            DateWindow { start: today - Duration::days(7), end: today }
                        } else {
                            DateWindow {
                                start: today - Duration::days(14),
                                end: today - Duration::days(6),
                            }
                        });
                    }
                    "weekend" => {
                        let saturday = previous_weekday(today, Weekday::Sat, !is_this);
                        return Some(DateWindow { start: saturday, end: saturday + Duration::days(1) });
                    }
                    "month" => {
                        return Some(if is_this {
                            DateWindow { start: today - Duration::days(30), end: today }
                        } else {
                            DateWindow {
                                start: today - Duration::days(45),
                                end: today - Duration::days(20),
                            }
                        });
                    }
                    "year" => {
                        return Some(if is_this {
                            DateWindow { start: today - Duration::days(365), end: today }
                        } else {
                            DateWindow {
                                start: today - Duration::days(730),
                                end: today - Duration::days(365),
                            }
                        });
                    }
                    "few" | "couple" => {
                        // `last few days/weeks`
                        let unit_index = if has(index + 2, "of") { index + 3 } else { index + 2 };
                        if let Some(unit) = tokens.get(unit_index) {
                            let span = match unit.as_str() {
                                "days" | "day" | "jours" => 4,
                                "weeks" | "week" | "semaines" => 21,
                                "months" | "month" | "mois" => 90,
                                _ => 0,
                            };
                            if span > 0 {
                                return Some(DateWindow { start: today - Duration::days(span), end: today });
                            }
                        }
                    }
                    other => {
                        if let Some(weekday) = weekday_of(other) {
                            let date = previous_weekday(today, weekday, !is_this);
                            return Some(DateWindow::around(date, 0));
                        }
                    }
                }
            }
        }

        // French: `la semaine dernière`, `le mois dernier`, `mardi dernier`
        if matches!(token, "dernière" | "derniere" | "dernier" | "passée" | "passee" | "passé") && index >= 1 {
            let previous = tokens[index - 1].as_str();
            match previous {
                "semaine" => {
                    return Some(DateWindow {
                        start: today - Duration::days(14),
                        end: today - Duration::days(6),
                    })
                }
                "mois" => {
                    return Some(DateWindow {
                        start: today - Duration::days(45),
                        end: today - Duration::days(20),
                    })
                }
                "année" | "annee" | "an" => {
                    return Some(DateWindow {
                        start: today - Duration::days(730),
                        end: today - Duration::days(365),
                    })
                }
                other => {
                    if let Some(weekday) = weekday_of(other) {
                        return Some(DateWindow::around(previous_weekday(today, weekday, true), 0));
                    }
                }
            }
        }

        // `in March` / `en mars` / `last March` — most recent occurrence not after today.
        if matches!(token, "in" | "en" | "during" | "since" | "depuis" | "back") || token == "last" {
            if let Some(month) = tokens.get(index + 1).and_then(|t| month_number(t)) {
                if tokens.get(index + 1).map(|t| t.len()).unwrap_or(0) >= 3 {
                    let year_hint = tokens.get(index + 2).and_then(|t| parse_year(t));
                    return Some(month_window(today, month, year_hint));
                }
            }
        }
    }

    // Bare month name anywhere (`what happened in early September`).
    for (index, token) in tokens.iter().enumerate() {
        if token.len() >= 4 {
            if let Some(month) = month_number(token) {
                let year_hint = tokens.get(index + 1).and_then(|t| parse_year(t));
                return Some(month_window(today, month, year_hint));
            }
        }
    }

    None
}

fn window_ago(today: NaiveDate, count: i64, unit: &str) -> Option<DateWindow> {
    let (days, slack) = match unit {
        "day" | "days" | "jour" | "jours" => (count, 1),
        "week" | "weeks" | "semaine" | "semaines" => (count * 7, 3),
        "month" | "months" | "mois" => (count * 30, 8),
        "year" | "years" | "an" | "ans" | "année" | "années" => (count * 365, 30),
        _ => return None,
    };
    Some(DateWindow::around(today - Duration::days(days), slack))
}

fn previous_weekday(today: NaiveDate, weekday: Weekday, strictly_before: bool) -> NaiveDate {
    let mut date = today;
    if strictly_before {
        date -= Duration::days(1);
    }
    while date.weekday() != weekday {
        date -= Duration::days(1);
    }
    date
}

fn month_window(today: NaiveDate, month: u32, year_hint: Option<i32>) -> DateWindow {
    let year = year_hint.unwrap_or_else(|| {
        if month <= today.month() {
            today.year()
        } else {
            today.year() - 1
        }
    });
    let start = NaiveDate::from_ymd_opt(year, month, 1).unwrap_or(today);
    let end = if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1)
    }
    .map(|next| next - Duration::days(1))
    .unwrap_or(start);
    DateWindow { start, end }
}

/// Multiplier applied to a candidate's score given the dates attached to
/// it. Inside the window: boost. Dated but far outside: mild demotion.
/// Undated or near the edge: neutral. The demotion is bounded so a
/// wrongly-parsed window never buries the right answer.
pub fn window_factor(window: &DateWindow, memory_dates: &[NaiveDate]) -> f64 {
    if memory_dates.is_empty() {
        return 1.0;
    }
    if memory_dates.iter().any(|date| window.contains(*date)) {
        return 1.25;
    }
    let nearest = memory_dates
        .iter()
        .map(|date| window.distance_days(*date))
        .min()
        .unwrap_or(i64::MAX);
    // Tolerance scales with the window: a one-day window tolerates ±2
    // days, a month tolerates about a week.
    let tolerance = (window.len_days() / 4).clamp(2, 10);
    if nearest <= tolerance {
        1.0
    } else {
        0.85
    }
}

/// The calendar day of an RFC 3339 / `YYYY-MM-DD…` timestamp.
pub fn day_of_timestamp(timestamp: &str) -> Option<NaiveDate> {
    let head = timestamp.get(..10)?;
    NaiveDate::parse_from_str(head, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(head, "%Y/%m/%d"))
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn extracts_explicit_dates_in_both_languages() {
        let today = d(2026, 9, 16);
        assert_eq!(extract_dates("deployed on 2026-09-01", None), vec![d(2026, 9, 1)]);
        assert_eq!(extract_dates("[date: 2023/05/20 (Sat) 02:21] user: hi", None), vec![d(2023, 5, 20)]);
        assert_eq!(extract_dates("réunion le 12 septembre 2026", None), vec![d(2026, 9, 12)]);
        assert_eq!(extract_dates("on September 12, 2026 we shipped", None), vec![d(2026, 9, 12)]);
        assert_eq!(extract_dates("le 3 mars", Some(today)), vec![d(2026, 3, 3)]);
        assert_eq!(extract_dates("livraison 16/09/2026", None), vec![d(2026, 9, 16)]);
        assert_eq!(extract_dates("hier j'ai publié", Some(today)), vec![d(2026, 9, 15)]);
        assert!(extract_dates("le 3 mars", None).is_empty());
        assert!(extract_dates("version 1.2.3 and PID 339", None).is_empty());
    }

    #[test]
    fn parses_relative_windows() {
        let today = d(2026, 9, 16); // a Wednesday
        let last_week = parse_query_window("what did I cook last week?", today).unwrap();
        assert_eq!(last_week.start, d(2026, 9, 2));
        assert_eq!(last_week.end, d(2026, 9, 10));

        let two_weeks = parse_query_window("the concert I went to two weeks ago", today).unwrap();
        assert!(two_weeks.contains(d(2026, 9, 2)));
        assert!(!two_weeks.contains(d(2026, 9, 14)));

        let tuesday = parse_query_window("the meeting last Tuesday", today).unwrap();
        assert_eq!(tuesday.start, d(2026, 9, 15));

        let fr = parse_query_window("qu'est-ce que j'ai fait la semaine dernière ?", today).unwrap();
        assert_eq!(fr, last_week);

        let il_y_a = parse_query_window("le déploiement il y a trois jours", today).unwrap();
        assert!(il_y_a.contains(d(2026, 9, 13)));

        let march = parse_query_window("what happened in March", today).unwrap();
        assert_eq!(march.start, d(2026, 3, 1));
        assert_eq!(march.end, d(2026, 3, 31));

        let explicit = parse_query_window("what did I do on 2026-09-01", today).unwrap();
        assert!(explicit.contains(d(2026, 9, 1)));

        assert!(parse_query_window("what is my favourite colour", today).is_none());
        assert!(parse_query_window("the last time I saw her", today).is_none());
    }

    #[test]
    fn window_factor_is_bounded() {
        let window = DateWindow { start: d(2026, 9, 2), end: d(2026, 9, 10) };
        assert_eq!(window_factor(&window, &[]), 1.0);
        assert_eq!(window_factor(&window, &[d(2026, 9, 5)]), 1.25);
        assert_eq!(window_factor(&window, &[d(2026, 9, 11)]), 1.0);
        assert_eq!(window_factor(&window, &[d(2026, 6, 1)]), 0.85);
    }

    #[test]
    fn day_of_timestamp_accepts_rfc3339() {
        assert_eq!(day_of_timestamp("2026-09-16T21:30:00+00:00"), Some(d(2026, 9, 16)));
        assert_eq!(day_of_timestamp("2023/05/20 (Sat) 02:21"), Some(d(2023, 5, 20)));
        assert_eq!(day_of_timestamp("nope"), None);
    }
}
