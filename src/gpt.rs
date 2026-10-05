use crate::tensor::*;
use crate::nn::*;

pub struct GPT {
    pub wte: T,
    pub blocks: Vec<Block>,
    pub ln_f_g: T, pub ln_f_b: T,
    pub lm_head: Linear,
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub rope_base: f32,
}

impl GPT {
    pub fn new(vocab_size: usize, block_size: usize, n_embd: usize, n_layer: usize, n_head: usize, seed: u64) -> Self {
        assert_eq!(n_embd % n_head, 0, "n_embd doit etre divisible par n_head");
        let wte = leaf(rand_init(vocab_size * n_embd, 0.02, seed), vec![vocab_size, n_embd]);
        let mut blocks = vec![];
        for i in 0..n_layer {
            blocks.push(Block::new(n_embd, n_head, seed + 100 + i as u64 * 10));
        }
        Self {
            wte, blocks,
            ln_f_g: leaf(vec![1.0; n_embd], vec![n_embd]),
            ln_f_b: leaf(vec![0.0; n_embd], vec![n_embd]),
            lm_head: Linear::new(n_embd, vocab_size, seed+999),
            vocab_size, block_size, n_embd, n_head,
            rope_base: 10000.0,
        }
    }

    pub fn forward(&self, idx: &[usize]) -> T {
        let tok = embedding_lookup(&self.wte, idx);
        let mut x = tok;
        for b in &self.blocks { x = b.forward(&x, 0); }
        let x = layernorm(&x, &self.ln_f_g, &self.ln_f_b, 1e-5);
        self.lm_head.forward(&x)
    }

    pub fn params(&self) -> Vec<T> {
        let mut p = vec![self.wte.clone()];
        for b in &self.blocks { p.extend(b.params()); }
        p.push(self.ln_f_g.clone());
        p.push(self.ln_f_b.clone());
        p.extend(self.lm_head.params());
        p
    }

    pub fn generate(&self, prompt: &[usize], n_new: usize, seed: &mut u64) -> Vec<usize> {
        let mut idx = prompt.to_vec();
        for _ in 0..n_new {
            let start = if idx.len() > self.block_size { idx.len() - self.block_size } else { 0 };
            let ctx: Vec<usize> = idx[start..].to_vec();
            let logits = self.forward(&ctx);
            let t = ctx.len();
            let v = self.vocab_size;
            let lb = logits.borrow();
            let last = &lb.data[(t-1)*v..t*v];
            let next = sample_from(last, seed);
            idx.push(next);
        }
        idx
    }

    pub fn generate_cached(&self, prompt: &[usize], n_new: usize, seed: &mut u64) -> Vec<usize> {
        let n_layer = self.blocks.len();
        let n_head = self.n_head;
        let mut cache_k: Vec<Vec<Vec<f32>>> = (0..n_layer)
            .map(|_| (0..n_head).map(|_| Vec::new()).collect()).collect();
        let mut cache_v: Vec<Vec<Vec<f32>>> = (0..n_layer)
            .map(|_| (0..n_head).map(|_| Vec::new()).collect()).collect();

        let mut idx = prompt.to_vec();
        let mut last_logits = vec![0.0f32; self.vocab_size];

        for (pos, &tok) in prompt.iter().enumerate() {
            last_logits = self.step_raw(tok, pos, &mut cache_k, &mut cache_v);
        }

        let mut pos = prompt.len();
        for _ in 0..n_new {
            let next = sample_from(&last_logits, seed);
            idx.push(next);
            last_logits = self.step_raw(next, pos, &mut cache_k, &mut cache_v);
            pos += 1;
        }
        idx
    }

    fn step_raw(
        &self,
        token: usize,
        pos: usize,
        cache_k: &mut Vec<Vec<Vec<f32>>>,
        cache_v: &mut Vec<Vec<Vec<f32>>>,
    ) -> Vec<f32> {
        let c = self.n_embd;
        let hd = c / self.n_head;

        let mut x = vec![0.0f32; c];
        {
            let wte = self.wte.borrow();
            for j in 0..c { x[j] = wte.data[token*c + j]; }
        }

        for (li, block) in self.blocks.iter().enumerate() {
            let xn = layernorm_raw(&x, &block.ln1_g.borrow().data, &block.ln1_b.borrow().data, 1, c, 1e-5);
            let mut q = linear_raw(&xn, &block.att.wq.w.borrow().data, &block.att.wq.b.borrow().data, c, c);
            let mut knew = linear_raw(&xn, &block.att.wk.w.borrow().data, &block.att.wk.b.borrow().data, c, c);
            let vnew = linear_raw(&xn, &block.att.wv.w.borrow().data, &block.att.wv.b.borrow().data, c, c);

            rope_apply_raw(&mut q,    hd, self.rope_base, pos);
            rope_apply_raw(&mut knew, hd, self.rope_base, pos);

            let mut att_concat = vec![0.0f32; c];
            for h in 0..self.n_head {
                if cache_k[li][h].len() / hd >= self.block_size {
                    cache_k[li][h].drain(0..hd);
                    cache_v[li][h].drain(0..hd);
                }
                for j in 0..hd {
                    cache_k[li][h].push(knew[h*hd + j]);
                    cache_v[li][h].push(vnew[h*hd + j]);
                }
                let t_prev = cache_k[li][h].len() / hd;
                let qh = &q[h*hd..(h+1)*hd];
                let mut scores = vec![0.0f32; t_prev];
                for i in 0..t_prev {
                    let mut s = 0.0;
                    for j in 0..hd { s += qh[j] * cache_k[li][h][i*hd + j]; }
                    scores[i] = s / (hd as f32).sqrt();
                }
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in &mut scores { *s = (*s - max).exp(); sum += *s; }
                for s in &mut scores { *s /= sum; }
                for j in 0..hd {
                    let mut acc = 0.0;
                    for i in 0..t_prev { acc += scores[i] * cache_v[li][h][i*hd + j]; }
                    att_concat[h*hd + j] = acc;
                }
            }

            let att_out = linear_raw(&att_concat,
                &block.att.wo.w.borrow().data,
                &block.att.wo.b.borrow().data, c, c);
            for j in 0..c { x[j] += att_out[j]; }

            let xn = layernorm_raw(&x, &block.ln2_g.borrow().data, &block.ln2_b.borrow().data, 1, c, 1e-5);
            let h1 = linear_raw(&xn, &block.ffn.w1.w.borrow().data, &block.ffn.w1.b.borrow().data, c, 4*c);
            let h1 = gelu_raw(&h1);
            let h2 = linear_raw(&h1, &block.ffn.w2.w.borrow().data, &block.ffn.w2.b.borrow().data, 4*c, c);
            for j in 0..c { x[j] += h2[j]; }
        }

        let xn = layernorm_raw(&x, &self.ln_f_g.borrow().data, &self.ln_f_b.borrow().data, 1, c, 1e-5);
        linear_raw(&xn, &self.lm_head.w.borrow().data, &self.lm_head.b.borrow().data, c, self.vocab_size)
    }
}

fn linear_raw(x: &[f32], w: &[f32], b: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let t = x.len() / n_in;
    let mut y = vec![0.0f32; t * n_out];
    for i in 0..t {
        for j in 0..n_out {
            let mut s = b[j];
            for k in 0..n_in { s += x[i*n_in + k] * w[k*n_out + j]; }
            y[i*n_out + j] = s;
        }
    }
    y
}

fn layernorm_raw(x: &[f32], g: &[f32], b: &[f32], t: usize, c: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; t*c];
    for i in 0..t {
        let mut mean = 0.0;
        for j in 0..c { mean += x[i*c+j]; }
        mean /= c as f32;
        let mut var = 0.0;
        for j in 0..c { let d = x[i*c+j] - mean; var += d*d; }
        var /= c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..c { out[i*c+j] = (x[i*c+j] - mean) * inv * g[j] + b[j]; }
    }
    out
}

fn gelu_raw(x: &[f32]) -> Vec<f32> {
    let sqrt2pi = (2.0f32 / std::f32::consts::PI).sqrt();
    x.iter().map(|&v| {
        let inner = sqrt2pi * (v + 0.044715 * v*v*v);
        0.5 * v * (1.0 + inner.tanh())
    }).collect()
}

fn sample_from(logits: &[f32], seed: &mut u64) -> usize {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|&x| (x-max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let r = ((*seed >> 33) as f32) / ((1u64 << 31) as f32);
    let mut cum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cum += p / sum;
        if cum > r { return i; }
    }
    logits.len() - 1
}

fn rope_apply_raw(x: &mut [f32], head_dim: usize, base: f32, pos: usize) {
    let c = x.len();
    let n_head = c / head_dim;
    for h in 0..n_head {
        let base_idx = h * head_dim;
        for j in 0..head_dim / 2 {
            let theta = pos as f32 / base.powf(2.0 * (j as f32) / (head_dim as f32));
            let (s, co) = theta.sin_cos();
            let a = x[base_idx + 2*j];
            let b = x[base_idx + 2*j + 1];
            x[base_idx + 2*j]     = a * co - b * s;
            x[base_idx + 2*j + 1] = a * s  + b * co;
        }
    }
}
