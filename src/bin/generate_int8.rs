use llm_rs::gpt::GPT;
use llm_rs::bpe_tokenizer::BpeTokenizer;
use std::path::Path;
use std::fs;

const MAGIC: &[u8; 8] = b"LLMRSINT";

fn load_int8(model: &GPT, path: &str) {
    let bytes = fs::read(path).expect("fichier int8 introuvable");
    assert_eq!(&bytes[0..8], MAGIC, "magic int8 inattendu");
    let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let params = model.params();
    assert_eq!(n, params.len(), "nb tenseurs incompatible");
    let mut off = 12;
    for p in &params {
        let len   = u32::from_le_bytes(bytes[off..off+4].try_into().unwrap()) as usize;
        off += 4;
        let scale = f32::from_le_bytes(bytes[off..off+4].try_into().unwrap());
        off += 4;
        let mut pb = p.borrow_mut();
        for j in 0..len {
            let q = bytes[off + j] as i8;
            pb.data[j] = q as f32 * scale;
        }
        off += len;
    }
    println!("Modele int8 dequantifie depuis {}", path);
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let vocab_size = 1024;
    let block_size = 128;
    let n_embd = 96;
    let n_layer = 4;
    let n_head = 8;

    let model = GPT::new(vocab_size, block_size, n_embd, n_layer, n_head, 42);
    load_int8(&model, root.join("models").join("gpt_int8.bin").to_str().unwrap());

    let tok = BpeTokenizer::load(
        root.join("models").join("tokenizer_bpe.json").to_str().unwrap()
    ).unwrap();

    let mut seed: u64 = 42;
    for p in ["The ", "It was ", "She said "] {
        let ids = tok.encode(p);
        let out = model.generate_cached(&ids, 150, &mut seed);
        println!("--- {:?} ---", p);
        println!("{}", tok.decode(&out));
        println!();
    }
}
