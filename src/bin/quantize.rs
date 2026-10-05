use std::path::Path;
use std::fs;

const MAGIC_IN:  &[u8; 8] = b"LLMRS003";
const MAGIC_OUT: &[u8; 8] = b"LLMRSINT";

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let in_path  = root.join("checkpoints").join("latest.bin");
    let out_path = root.join("models").join("gpt_int8.bin");

    let bytes = fs::read(&in_path).expect("checkpoint introuvable");
    assert_eq!(&bytes[0..8], MAGIC_IN, "magic inattendu");

    let n_tensors = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    let mut off = 24;

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(MAGIC_OUT);
    out.extend_from_slice(&(n_tensors as u32).to_le_bytes());

    let mut orig_total = 0usize;
    let mut quant_total = 0usize;

    for _ in 0..n_tensors {
        let len = u32::from_le_bytes(bytes[off..off+4].try_into().unwrap()) as usize;
        off += 4;
        let mut v = Vec::with_capacity(len);
        let mut max_abs = 0.0f32;
        for _ in 0..len {
            let f = f32::from_le_bytes(bytes[off..off+4].try_into().unwrap());
            off += 4;
            if f.abs() > max_abs { max_abs = f.abs(); }
            v.push(f);
        }
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let inv = 1.0 / scale;
        out.extend_from_slice(&(len as u32).to_le_bytes());
        out.extend_from_slice(&scale.to_le_bytes());
        for f in &v {
            let q = (f * inv).round().clamp(-127.0, 127.0) as i8;
            out.push(q as u8);
        }
        orig_total += len * 4;
        quant_total += len + 8;
    }

    fs::write(&out_path, &out).unwrap();
    println!("Ecrit : {}", out_path.display());
    println!("Original : {} octets ({:.2} MB)", orig_total, orig_total as f64 / 1e6);
    println!("Int8     : {} octets ({:.2} MB)", quant_total, quant_total as f64 / 1e6);
    println!("Ratio    : x{:.2}", orig_total as f64 / quant_total as f64);
}
