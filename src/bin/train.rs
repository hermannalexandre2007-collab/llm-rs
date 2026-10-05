use llm_rs::gpt::GPT;
use llm_rs::tensor::{backward, cross_entropy, zero_grad, T};
use llm_rs::train_utils::{load_corpus, AdamW};
use llm_rs::bpe_tokenizer::BpeTokenizer;
use std::path::Path;
use std::fs;

const MAGIC: &[u8; 8] = b"LLMRS003";

fn save_checkpoint(model: &GPT, path: &str, step: usize, rng: u64) {
    let params = model.params();
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(step as u32).to_le_bytes());
    bytes.extend_from_slice(&rng.to_le_bytes());
    bytes.extend_from_slice(&(params.len() as u32).to_le_bytes());
    for p in &params {
        let pb = p.borrow();
        bytes.extend_from_slice(&(pb.data.len() as u32).to_le_bytes());
        for &f in &pb.data { bytes.extend_from_slice(&f.to_le_bytes()); }
    }
    fs::write(path, bytes).unwrap();
    let total: usize = params.iter().map(|p| p.borrow().data.len()).sum();
    println!("Checkpoint step {} : {} params", step, total);
}

fn load_checkpoint(model: &GPT, path: &str) -> Option<(usize, u64)> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 24 || &bytes[0..8] != MAGIC { return None; }
    let step = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let rng = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    let n = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    let params = model.params();
    if n != params.len() { return None; }
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
    Some((step, rng))
}

fn clip_grads(params: &[T], max_norm: f32) {
    let mut sq = 0.0f32;
    for p in params { for &g in &p.borrow().grad { sq += g * g; } }
    let norm = sq.sqrt();
    if norm > max_norm {
        let s = max_norm / (norm + 1e-6);
        for p in params {
            for g in p.borrow_mut().grad.iter_mut() { *g *= s; }
        }
    }
}

fn lr_at(step: usize, max_steps: usize, base_lr: f32, warmup: usize) -> f32 {
    if step < warmup {
        base_lr * (step + 1) as f32 / warmup as f32
    } else {
        let p = (step - warmup) as f32 / (max_steps - warmup) as f32;
        let min_lr = base_lr * 0.1;
        min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * p).cos())
    }
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();

    println!("Chargement du corpus...");
    let data = load_corpus(&root);
    let n = data.len();
    let split = (n * 9) / 10;
    let train = &data[..split];
    let val = &data[split..];
    println!("Train : {} | Val : {}", train.len(), val.len());

    let tok = BpeTokenizer::load(
        root.join("models").join("tokenizer_bpe.json").to_str().unwrap()
    ).unwrap();

    let vocab_size = 1024;
    let block_size = 128;
    let n_embd = 96;
    let n_layer = 4;
    let n_head = 8;

    println!("GPT : vocab={} block={} embd={} layer={} head={} RoPE",
        vocab_size, block_size, n_embd, n_layer, n_head);

    let model = GPT::new(vocab_size, block_size, n_embd, n_layer, n_head, 42);
    let params = model.params();
    let n_params: usize = params.iter().map(|p| p.borrow().data.len()).sum();
    println!("Parametres : {}", n_params);

    fs::create_dir_all(root.join("checkpoints")).unwrap();
    let ckpt_path = root.join("checkpoints").join("latest.bin");
    let ckpt_path = ckpt_path.to_str().unwrap();

    let base_lr = 1e-3;
    let max_steps = 40000;
    let warmup = 200;
    let batch_size = 1;

    let mut opt = AdamW::new(&params, base_lr);
    let mut rng: u64 = 12345;
    let mut start_step = 0;

    if let Some((step, saved_rng)) = load_checkpoint(&model, ckpt_path) {
        println!("Reprise depuis checkpoint step {}", step);
        start_step = step;
        rng = saved_rng;
    } else {
        let _ = fs::remove_file(ckpt_path);
        println!("Nouveau checkpoint (pas de reprise)");
    }

    let rand_range = |lo: usize, hi: usize, s: &mut u64| -> usize {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        lo + ((*s >> 33) as usize) % (hi - lo)
    };

    println!("Entrainement {} -> {} steps, batch={}, lr={}", start_step, max_steps, batch_size, base_lr);

    for step in start_step..max_steps {
        zero_grad(&params);
        let mut loss_sum = 0.0f32;
        for _ in 0..batch_size {
            let s0 = rand_range(0, train.len() - block_size - 1, &mut rng);
            let x: Vec<usize> = train[s0..s0+block_size].to_vec();
            let y: Vec<usize> = train[s0+1..s0+block_size+1].to_vec();
            let logits = model.forward(&x);
            let loss = cross_entropy(&logits, &y);
            loss_sum += loss.borrow().data[0];
            backward(&loss);
        }
        for p in &params {
            let mut pb = p.borrow_mut();
            for g in pb.grad.iter_mut() { *g /= batch_size as f32; }
        }
        clip_grads(&params, 1.0);
        opt.lr = lr_at(step, max_steps, base_lr, warmup);
        opt.step(&params);

        if step % 200 == 0 || step == max_steps - 1 {
            let mut val_loss = 0.0f32;
            let vb = 4;
            for _ in 0..vb {
                let s0 = rand_range(0, val.len() - block_size - 1, &mut rng);
                let x: Vec<usize> = val[s0..s0+block_size].to_vec();
                let y: Vec<usize> = val[s0+1..s0+block_size+1].to_vec();
                let logits = model.forward(&x);
                let loss = cross_entropy(&logits, &y);
                val_loss += loss.borrow().data[0];
            }
            val_loss /= vb as f32;
            println!("step {:4}  train {:.4}  val {:.4}  lr {:.5}",
                step, loss_sum / batch_size as f32, val_loss, opt.lr);
        }

        if (step + 1) % 500 == 0 { save_checkpoint(&model, ckpt_path, step + 1, rng); }
    }

    save_checkpoint(&model, ckpt_path, max_steps, rng);
    fs::copy(ckpt_path, root.join("models").join("gpt.bin")).unwrap();

    println!("\n--- Generation KV cache ---");
    for p in ["The ", "It was ", "She ", "Mr. "] {
        let ids = tok.encode(p);
        let mut s: u64 = 999;
        let t0 = std::time::Instant::now();
        let out = model.generate_cached(&ids, 150, &mut s);
        let dt = t0.elapsed();
        println!("--- {:?}  ({:?}) ---", p, dt);
        println!("{}", tok.decode(&out));
        println!();
    }
}
