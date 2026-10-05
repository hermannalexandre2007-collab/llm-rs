use llm_rs::gpt::GPT;
use llm_rs::bpe_tokenizer::BpeTokenizer;
use std::path::Path;
use std::fs;

const MAGIC: &[u8; 8] = b"LLMRS003";

fn load_model(model: &GPT, path: &str) {
    let bytes = fs::read(path).expect("checkpoint introuvable");
    assert_eq!(&bytes[0..8], MAGIC, "magic mismatch");
    let n = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    let params = model.params();
    assert_eq!(n, params.len(), "nb tenseurs incompatible");
    let mut off = 24;
    for p in &params {
        let len = u32::from_le_bytes(bytes[off..off+4].try_into().unwrap()) as usize;
        off += 4;
        let mut pb = p.borrow_mut();
        for j in 0..len {
            pb.data[j] = f32::from_le_bytes(bytes[off..off+4].try_into().unwrap());
            off += 4;
        }
    }
    println!("Modele charge depuis {}", path);
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();

    let vocab_size = 1024;
    let block_size = 128;
    let n_embd = 96;
    let n_layer = 4;
    let n_head = 8;

    let model = GPT::new(vocab_size, block_size, n_embd, n_layer, n_head, 42);
    load_model(&model, root.join("checkpoints").join("latest.bin").to_str().unwrap());

    let tok = BpeTokenizer::load(
        root.join("models").join("tokenizer_bpe.json").to_str().unwrap()
    ).unwrap();

    let prompts = ["The ", "It was ", "She said ", "Mr. Darcy ", "Once upon a time"];
    let mut seed: u64 = 42;
    for p in prompts {
        let ids = tok.encode(p);
        let t0 = std::time::Instant::now();
        let out = model.generate_cached(&ids, 200, &mut seed);
        let dt = t0.elapsed();
        println!("--- {:?}  ({:?}) ---", p, dt);
        println!("{}", tok.decode(&out));
        println!();
    }
}
