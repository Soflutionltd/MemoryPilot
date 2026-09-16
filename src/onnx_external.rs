//! Move an ONNX model's weights out of the protobuf into a sidecar file.
//!
//! ONNX Runtime 1.28 keeps two to three copies of every initializer that
//! is embedded in the `.onnx` protobuf while a session loads (measured:
//! a 118 MB int8 cross-encoder costs +305 MB resident, and the memory is
//! never handed back). Weights stored as *external data* are memory-
//! mapped instead: the same model then costs +4 MB at load and only the
//! pages actually touched afterwards — pages the OS can drop under
//! pressure because they are file-backed.
//!
//! Hugging Face repositories are inconsistent about which layout they
//! ship, so MemoryPilot converts on the user's machine, once, right
//! after download. The conversion is a byte-level rewrite of the
//! protobuf wire format: every field that is not a large `raw_data`
//! initializer is copied verbatim, so the graph itself is untouched
//! (asserted by comparing session outputs in the tests).
//!
//! Wire-format reference (only what we touch):
//! - `ModelProto.graph`            = field 7  (message)
//! - `GraphProto.initializer`      = field 5  (repeated `TensorProto`)
//! - `TensorProto.raw_data`        = field 9  (bytes)
//! - `TensorProto.external_data`   = field 13 (repeated `StringStringEntryProto`)
//! - `TensorProto.data_location`   = field 14 (enum, `EXTERNAL` = 1)
//! - `StringStringEntryProto.key`  = field 1, `.value` = field 2

use std::io::Write;
use std::path::Path;

const WIRE_VARINT: u8 = 0;
const WIRE_FIXED64: u8 = 1;
const WIRE_LEN: u8 = 2;
const WIRE_FIXED32: u8 = 5;

const MODEL_GRAPH: u32 = 7;
const GRAPH_INITIALIZER: u32 = 5;
const TENSOR_RAW_DATA: u32 = 9;
const TENSOR_EXTERNAL_DATA: u32 = 13;
const TENSOR_DATA_LOCATION: u32 = 14;
const DATA_LOCATION_EXTERNAL: u64 = 1;

/// Tensors smaller than this stay inline: thousands of tiny scale /
/// zero-point scalars are cheaper in the graph than as mmap entries.
const EXTERNAL_THRESHOLD: usize = 1024;
/// Offsets in the sidecar are aligned so every tensor starts on a cache
/// line; ONNX Runtime does not require it, the CPU appreciates it.
const ALIGN: usize = 64;

/// One decoded protobuf field: tag, wire type and the raw payload bytes
/// (varint bytes, fixed bytes, or the length-delimited content).
struct Field<'a> {
    number: u32,
    wire: u8,
    payload: &'a [u8],
}

fn read_varint(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = *bytes
            .get(*cursor)
            .ok_or("protobuf: truncated varint")?;
        *cursor += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift > 63 {
            return Err("protobuf: varint too long".into());
        }
    }
}

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn write_tag(out: &mut Vec<u8>, number: u32, wire: u8) {
    write_varint(out, (u64::from(number) << 3) | u64::from(wire));
}

fn write_len_field(out: &mut Vec<u8>, number: u32, payload: &[u8]) {
    write_tag(out, number, WIRE_LEN);
    write_varint(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

/// Split a message into its fields without interpreting them.
fn fields(bytes: &[u8]) -> Result<Vec<Field<'_>>, String> {
    let mut cursor = 0;
    let mut out = Vec::new();
    while cursor < bytes.len() {
        let tag = read_varint(bytes, &mut cursor)?;
        let number = (tag >> 3) as u32;
        let wire = (tag & 7) as u8;
        let start = cursor;
        let payload = match wire {
            WIRE_VARINT => {
                read_varint(bytes, &mut cursor)?;
                &bytes[start..cursor]
            }
            WIRE_FIXED64 => {
                cursor += 8;
                bytes.get(start..cursor).ok_or("protobuf: truncated fixed64")?
            }
            WIRE_FIXED32 => {
                cursor += 4;
                bytes.get(start..cursor).ok_or("protobuf: truncated fixed32")?
            }
            WIRE_LEN => {
                let len = read_varint(bytes, &mut cursor)? as usize;
                let end = cursor
                    .checked_add(len)
                    .filter(|end| *end <= bytes.len())
                    .ok_or("protobuf: truncated length-delimited field")?;
                cursor = end;
                &bytes[end - len..end]
            }
            other => return Err(format!("protobuf: unsupported wire type {}", other)),
        };
        out.push(Field {
            number,
            wire,
            payload,
        });
    }
    Ok(out)
}

fn write_field(out: &mut Vec<u8>, field: &Field<'_>) {
    write_tag(out, field.number, field.wire);
    if field.wire == WIRE_LEN {
        write_varint(out, field.payload.len() as u64);
    }
    out.extend_from_slice(field.payload);
}

fn string_entry(key: &str, value: &str) -> Vec<u8> {
    let mut entry = Vec::with_capacity(key.len() + value.len() + 4);
    write_len_field(&mut entry, 1, key.as_bytes());
    write_len_field(&mut entry, 2, value.as_bytes());
    entry
}

struct Sidecar<'a> {
    file: std::fs::File,
    name: &'a str,
    offset: usize,
}

impl Sidecar<'_> {
    fn append(&mut self, data: &[u8]) -> Result<usize, String> {
        let padding = (ALIGN - self.offset % ALIGN) % ALIGN;
        if padding > 0 {
            self.file
                .write_all(&[0u8; ALIGN][..padding])
                .map_err(|error| format!("sidecar write: {}", error))?;
            self.offset += padding;
        }
        let at = self.offset;
        self.file
            .write_all(data)
            .map_err(|error| format!("sidecar write: {}", error))?;
        self.offset += data.len();
        Ok(at)
    }
}

/// Rewrite one `TensorProto`: large `raw_data` goes to the sidecar and is
/// replaced by `data_location = EXTERNAL` + location/offset/length
/// entries. Everything else is copied as-is.
fn externalize_tensor(tensor: &[u8], sidecar: &mut Sidecar<'_>) -> Result<Vec<u8>, String> {
    let parsed = fields(tensor)?;
    let raw = parsed
        .iter()
        .find(|field| field.number == TENSOR_RAW_DATA && field.wire == WIRE_LEN)
        .map(|field| field.payload);
    let Some(raw) = raw.filter(|raw| raw.len() >= EXTERNAL_THRESHOLD) else {
        return Ok(tensor.to_vec());
    };
    let offset = sidecar.append(raw)?;
    let mut out = Vec::with_capacity(tensor.len() - raw.len() + 96);
    for field in &parsed {
        if field.number == TENSOR_RAW_DATA {
            continue;
        }
        write_field(&mut out, field);
    }
    write_tag(&mut out, TENSOR_DATA_LOCATION, WIRE_VARINT);
    write_varint(&mut out, DATA_LOCATION_EXTERNAL);
    for (key, value) in [
        ("location", sidecar.name.to_string()),
        ("offset", offset.to_string()),
        ("length", raw.len().to_string()),
    ] {
        write_len_field(&mut out, TENSOR_EXTERNAL_DATA, &string_entry(key, &value));
    }
    Ok(out)
}

fn externalize_graph(graph: &[u8], sidecar: &mut Sidecar<'_>) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(graph.len() / 4);
    for field in fields(graph)? {
        if field.number == GRAPH_INITIALIZER && field.wire == WIRE_LEN {
            let tensor = externalize_tensor(field.payload, sidecar)?;
            write_len_field(&mut out, GRAPH_INITIALIZER, &tensor);
        } else {
            write_field(&mut out, &field);
        }
    }
    Ok(out)
}

/// Convert `source` (an `.onnx` file with embedded weights) into
/// `graph_out` + `data_out`, the latter named `data_name` inside the
/// graph so ONNX Runtime resolves it next to `graph_out`. Writes go to
/// temporary files first and are renamed at the end, so a crash midway
/// never leaves a half-written model that would be mistaken for a cache
/// hit.
pub fn externalize_model(source: &Path, graph_out: &Path, data_out: &Path) -> Result<(), String> {
    let data_name = data_out
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("sidecar path has no file name")?;
    let model = std::fs::read(source)
        .map_err(|error| format!("read {}: {}", source.display(), error))?;

    if let Some(parent) = graph_out.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("mkdir {}: {}", parent.display(), error))?;
    }
    let data_tmp = data_out.with_extension("onnx_data.part");
    let graph_tmp = graph_out.with_extension("onnx.part");
    let file = std::fs::File::create(&data_tmp)
        .map_err(|error| format!("create {}: {}", data_tmp.display(), error))?;
    let mut sidecar = Sidecar {
        file,
        name: data_name,
        offset: 0,
    };

    let mut out = Vec::with_capacity(model.len() / 4);
    for field in fields(&model)? {
        if field.number == MODEL_GRAPH && field.wire == WIRE_LEN {
            let graph = externalize_graph(field.payload, &mut sidecar)?;
            write_len_field(&mut out, MODEL_GRAPH, &graph);
        } else {
            write_field(&mut out, &field);
        }
    }
    sidecar
        .file
        .sync_all()
        .map_err(|error| format!("sidecar sync: {}", error))?;
    drop(sidecar);
    std::fs::write(&graph_tmp, &out)
        .map_err(|error| format!("write {}: {}", graph_tmp.display(), error))?;
    std::fs::rename(&data_tmp, data_out)
        .map_err(|error| format!("rename {}: {}", data_out.display(), error))?;
    std::fs::rename(&graph_tmp, graph_out)
        .map_err(|error| format!("rename {}: {}", graph_out.display(), error))?;
    Ok(())
}

/// Path of the externalized copy of `repo/file`, converting it on first
/// use. Lives under the fastembed cache so `FASTEMBED_CACHE_PATH` moves
/// it along with the downloads.
pub fn externalized(repo: &str, file: &str, show_progress: bool) -> Result<std::path::PathBuf, String> {
    let stem = Path::new(file)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| format!("bad onnx file name {}", file))?;
    let dir = crate::embedding::fastembed_cache_dir()
        .join("memorypilot-external")
        .join(repo.replace('/', "--"));
    let graph = dir.join(format!("{}.onnx", stem));
    let data = dir.join(format!("{}.onnx_data", stem));
    if graph.is_file() && data.is_file() {
        return Ok(graph);
    }
    let source = crate::embedding::hf_file(repo, file, show_progress)?;
    externalize_model(&source, &graph, &data)?;
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        for value in [0u64, 1, 127, 128, 300, 1 << 32, u64::MAX] {
            let mut buf = Vec::new();
            write_varint(&mut buf, value);
            let mut cursor = 0;
            assert_eq!(read_varint(&buf, &mut cursor).unwrap(), value);
            assert_eq!(cursor, buf.len());
        }
    }

    #[test]
    fn small_tensors_and_unknown_fields_are_copied_verbatim() {
        // ModelProto { ir_version(1)=9, graph(7) = GraphProto {
        //   initializer(5) = TensorProto { name(8)="w", raw_data(9)=[1,2,3] },
        //   doc_string(10) = "hello" } }
        let mut tensor = Vec::new();
        write_len_field(&mut tensor, 8, b"w");
        write_len_field(&mut tensor, 9, &[1, 2, 3]);
        let mut graph = Vec::new();
        write_len_field(&mut graph, 5, &tensor);
        write_len_field(&mut graph, 10, b"hello");
        let mut model = Vec::new();
        write_tag(&mut model, 1, WIRE_VARINT);
        write_varint(&mut model, 9);
        write_len_field(&mut model, 7, &graph);

        let dir = std::env::temp_dir().join(format!("mp-onnx-ext-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let src = dir.join("tiny.onnx");
        std::fs::write(&src, &model).unwrap();
        let graph_out = dir.join("out.onnx");
        let data_out = dir.join("out.onnx_data");
        externalize_model(&src, &graph_out, &data_out).unwrap();
        assert_eq!(std::fs::read(&graph_out).unwrap(), model, "below threshold: byte-identical");
        assert_eq!(std::fs::metadata(&data_out).unwrap().len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn large_tensor_moves_to_sidecar_with_external_entries() {
        let weights: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        let mut tensor = Vec::new();
        write_len_field(&mut tensor, 8, b"big");
        write_len_field(&mut tensor, 9, &weights);
        let mut graph = Vec::new();
        write_len_field(&mut graph, 5, &tensor);
        let mut model = Vec::new();
        write_len_field(&mut model, 7, &graph);

        let dir = std::env::temp_dir().join(format!("mp-onnx-ext-big-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let src = dir.join("big.onnx");
        std::fs::write(&src, &model).unwrap();
        let graph_out = dir.join("big_ext.onnx");
        let data_out = dir.join("big_ext.onnx_data");
        externalize_model(&src, &graph_out, &data_out).unwrap();

        assert_eq!(std::fs::read(&data_out).unwrap(), weights);
        let rewritten = std::fs::read(&graph_out).unwrap();
        assert!(rewritten.len() < 200, "weights must have left the graph");
        let graph_field = &fields(&rewritten).unwrap()[0];
        let tensor_field = &fields(graph_field.payload).unwrap()[0];
        let tensor_fields = fields(tensor_field.payload).unwrap();
        assert!(tensor_fields.iter().all(|f| f.number != TENSOR_RAW_DATA));
        let location = tensor_fields
            .iter()
            .find(|f| f.number == TENSOR_DATA_LOCATION)
            .expect("data_location");
        assert_eq!(location.payload, &[DATA_LOCATION_EXTERNAL as u8]);
        let entries: Vec<(String, String)> = tensor_fields
            .iter()
            .filter(|f| f.number == TENSOR_EXTERNAL_DATA)
            .map(|f| {
                let kv = fields(f.payload).unwrap();
                (
                    String::from_utf8(kv[0].payload.to_vec()).unwrap(),
                    String::from_utf8(kv[1].payload.to_vec()).unwrap(),
                )
            })
            .collect();
        assert_eq!(
            entries,
            vec![
                ("location".into(), "big_ext.onnx_data".into()),
                ("offset".into(), "0".into()),
                ("length".into(), "5000".into()),
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
