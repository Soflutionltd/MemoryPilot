//! Scoring head of a sentence-transformers cross-encoder whose ONNX graph
//! stops at the encoder (the Ettin rerankers): the repository ships the
//! last three modules apart, as small safetensors files.
//!
//! `[CLS]` vector → Dense (no bias) → GELU → LayerNorm → Dense(→1).

use std::collections::HashMap;
use std::path::Path;

/// `torch.nn.LayerNorm` default.
const LAYER_NORM_EPS: f64 = 1e-5;

pub struct ClsDenseHead {
    width: usize,
    /// `width × width`, row-major: one row per output unit.
    dense: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    out_weight: Vec<f32>,
    out_bias: f32,
}

impl ClsDenseHead {
    /// Reads `2_Dense`, `3_LayerNorm` and `4_Dense` of `repo`.
    pub fn load(repo: &str) -> Result<Self, String> {
        let module = |name: &str| -> Result<HashMap<String, Vec<f32>>, String> {
            let file = format!("{}/model.safetensors", name);
            let path = crate::embedding::hf_file(repo, &file, false)?;
            read_f32_tensors(&path)
        };
        let mut dense = module("2_Dense")?;
        let mut norm = module("3_LayerNorm")?;
        let mut out = module("4_Dense")?;
        let take = |tensors: &mut HashMap<String, Vec<f32>>, name: &str| {
            tensors
                .remove(name)
                .ok_or_else(|| format!("{}: tensor {} is missing", repo, name))
        };
        let out_bias = take(&mut out, "linear.bias")?;
        Self::from_parts(
            take(&mut dense, "linear.weight")?,
            take(&mut norm, "norm.weight")?,
            take(&mut norm, "norm.bias")?,
            take(&mut out, "linear.weight")?,
            out_bias.first().copied().ok_or("4_Dense bias is empty")?,
        )
    }

    fn from_parts(
        dense: Vec<f32>,
        norm_weight: Vec<f32>,
        norm_bias: Vec<f32>,
        out_weight: Vec<f32>,
        out_bias: f32,
    ) -> Result<Self, String> {
        let width = out_weight.len();
        if width == 0
            || dense.len() != width * width
            || norm_weight.len() != width
            || norm_bias.len() != width
        {
            return Err(format!(
                "scoring head shapes do not agree: dense {}, norm {} / {}, output {}",
                dense.len(),
                norm_weight.len(),
                norm_bias.len(),
                width
            ));
        }
        Ok(Self {
            width,
            dense,
            norm_weight,
            norm_bias,
            out_weight,
            out_bias,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    /// Relevance score of one pair from its `[CLS]` hidden state.
    pub fn score(&self, cls: &[f32]) -> Result<f32, String> {
        if cls.len() != self.width {
            return Err(format!("[CLS] vector has {} values, the head expects {}", cls.len(), self.width));
        }
        let activated: Vec<f64> = self
            .dense
            .chunks_exact(self.width)
            .map(|row| {
                let sum: f64 = row.iter().zip(cls).map(|(w, x)| *w as f64 * *x as f64).sum();
                gelu(sum)
            })
            .collect();
        let mean = activated.iter().sum::<f64>() / self.width as f64;
        let variance = activated.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / self.width as f64;
        let deviation = (variance + LAYER_NORM_EPS).sqrt();
        let score: f64 = activated
            .iter()
            .zip(&self.norm_weight)
            .zip(&self.norm_bias)
            .zip(&self.out_weight)
            .map(|(((value, gain), bias), weight)| {
                ((value - mean) / deviation * *gain as f64 + *bias as f64) * *weight as f64
            })
            .sum();
        Ok((score + self.out_bias as f64) as f32)
    }
}

/// Exact GELU, `x · Φ(x)`.
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Abramowitz & Stegun 7.1.26, absolute error below 1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let polynomial = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

/// The F32 tensors of a safetensors file, by name: an 8-byte little-endian
/// header length, a JSON header, then the raw little-endian values.
fn read_f32_tensors(path: &Path) -> Result<HashMap<String, Vec<f32>>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {}", path.display(), error))?;
    parse_f32_tensors(&bytes).map_err(|error| format!("{}: {}", path.display(), error))
}

fn parse_f32_tensors(bytes: &[u8]) -> Result<HashMap<String, Vec<f32>>, String> {
    #[derive(serde::Deserialize)]
    struct Entry {
        dtype: String,
        data_offsets: (usize, usize),
    }

    let header_len = bytes
        .get(..8)
        .and_then(|prefix| <[u8; 8]>::try_from(prefix).ok())
        .map(u64::from_le_bytes)
        .ok_or("safetensors file is shorter than its length prefix")? as usize;
    let header = bytes
        .get(8..8usize.saturating_add(header_len))
        .ok_or("safetensors header runs past the end of the file")?;
    let data = &bytes[8 + header_len..];
    let entries: HashMap<String, serde_json::Value> =
        serde_json::from_slice(header).map_err(|error| format!("safetensors header: {}", error))?;

    let mut tensors = HashMap::new();
    for (name, value) in entries {
        if name == "__metadata__" {
            continue;
        }
        let entry: Entry = serde_json::from_value(value)
            .map_err(|error| format!("safetensors entry {}: {}", name, error))?;
        if entry.dtype != "F32" {
            return Err(format!("tensor {} is {}, expected F32", name, entry.dtype));
        }
        let (start, end) = entry.data_offsets;
        let raw = data
            .get(start..end)
            .filter(|raw| raw.len() % 4 == 0)
            .ok_or_else(|| format!("tensor {} has bad offsets {}..{}", name, start, end))?;
        let values = raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        tensors.insert(name, values);
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safetensors(header: &str, values: &[f32]) -> Vec<u8> {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn safetensors_f32_tensors_are_read_by_name() {
        let header = r#"{"__metadata__":{"format":"pt"},"linear.bias":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"linear.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[4,12]}}"#;
        let tensors = parse_f32_tensors(&safetensors(header, &[0.5, 1.0, -2.0])).expect("parse");
        assert_eq!(tensors["linear.bias"], vec![0.5]);
        assert_eq!(tensors["linear.weight"], vec![1.0, -2.0]);
    }

    #[test]
    fn safetensors_errors_are_reported() {
        assert!(parse_f32_tensors(&[1, 2, 3]).is_err());
        let half = r#"{"w":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#;
        assert!(parse_f32_tensors(&safetensors(half, &[1.0])).unwrap_err().contains("F16"));
        let past = r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        assert!(parse_f32_tensors(&safetensors(past, &[1.0])).unwrap_err().contains("offsets"));
    }

    #[test]
    fn gelu_matches_reference_values() {
        // torch.nn.functional.gelu
        for (x, expected) in [(0.0, 0.0), (1.0, 0.841_344_7), (-1.0, -0.158_655_3), (3.0, 2.995_950_4)] {
            assert!((gelu(x) - expected).abs() < 1e-6, "gelu({}) = {}", x, gelu(x));
        }
    }

    #[test]
    fn head_scores_dense_gelu_norm_dense() {
        // Identity dense on [1, -1]: GELU gives [0.8413447, -0.1586553],
        // LayerNorm (gain 1, bias 0) gives [1, -1] up to eps, and the
        // output layer 2·1 + 3·(−1) + 0.5 = −0.5.
        let head = ClsDenseHead::from_parts(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![2.0, 3.0],
            0.5,
        )
        .expect("head");
        let score = head.score(&[1.0, -1.0]).expect("score");
        assert!((score + 0.5).abs() < 1e-3, "score {}", score);
        assert!(head.score(&[1.0]).is_err());
        assert!(ClsDenseHead::from_parts(vec![1.0; 3], vec![1.0; 2], vec![0.0; 2], vec![1.0; 2], 0.0).is_err());
    }
}
