use crate::tensor::*;
use std::fs;
use std::path::Path;
use crate::bpe_tokenizer::BpeTokenizer;

pub struct AdamW {
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
    pub t: u32,
}

impl AdamW {
    pub fn new(params: &[T], lr: f32) -> Self {
        let m = params.iter().map(|p| vec![0.0; p.borrow().data.len()]).collect();
        let v = params.iter().map(|p| vec![0.0; p.borrow().data.len()]).collect();
        Self { m, v, lr, beta1: 0.9, beta2: 0.95, eps: 1e-8, wd: 0.01, t: 0 }
    }
    pub fn step(&mut self, params: &[T]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for (i, p) in params.iter().enumerate() {
            let mut pb = p.borrow_mut();
            for j in 0..pb.data.len() {
                let g = pb.grad[j];
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
                let mh = self.m[i][j] / bc1;
                let vh = self.v[i][j] / bc2;
                let upd = mh / (vh.sqrt() + self.eps);
                pb.data[j] -= self.lr * (upd + self.wd * pb.data[j]);
            }
        }
    }
}

pub fn load_corpus(root: &Path) -> Vec<usize> {
    let text = fs::read_to_string(root.join("data").join("corpus_gutenberg.txt")).unwrap();
    let tok = BpeTokenizer::load(
        root.join("models").join("tokenizer_bpe.json").to_str().unwrap()
    ).unwrap();
    tok.encode(&text)
}
