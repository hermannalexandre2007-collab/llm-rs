use crate::tensor::*;

pub struct Linear { pub w: T, pub b: T }

impl Linear {
    pub fn new(n_in: usize, n_out: usize, seed: u64) -> Self {
        let scale = (1.0 / n_in as f32).sqrt();
        Self {
            w: leaf(rand_init(n_in * n_out, scale, seed), vec![n_in, n_out]),
            b: leaf(vec![0.0; n_out], vec![n_out]),
        }
    }
    pub fn forward(&self, x: &T) -> T {
        let y = matmul(x, &self.w);
        add_row(&y, &self.b)
    }
    pub fn params(&self) -> Vec<T> { vec![self.w.clone(), self.b.clone()] }
}

pub struct Attention {
    pub wq: Linear, pub wk: Linear, pub wv: Linear, pub wo: Linear,
    pub n_head: usize,
}

impl Attention {
    pub fn new(c: usize, n_head: usize, seed: u64) -> Self {
        Self {
            wq: Linear::new(c, c, seed+1),
            wk: Linear::new(c, c, seed+2),
            wv: Linear::new(c, c, seed+3),
            wo: Linear::new(c, c, seed+4),
            n_head,
        }
    }
    pub fn forward(&self, x: &T, pos_offset: usize) -> T {
        let c = x.borrow().shape[1];
        let hd = c / self.n_head;
        let q = self.wq.forward(x);
        let k = self.wk.forward(x);
        let v = self.wv.forward(x);
        let q = rope_apply(&q, hd, 10000.0, pos_offset);
        let k = rope_apply(&k, hd, 10000.0, pos_offset);

        let mut heads = Vec::with_capacity(self.n_head);
        for h in 0..self.n_head {
            let qh = slice_cols(&q, h*hd, hd);
            let kh = slice_cols(&k, h*hd, hd);
            let vh = slice_cols(&v, h*hd, hd);
            let kt = transpose(&kh);
            let att = matmul(&qh, &kt);
            let att = scale(&att, 1.0 / (hd as f32).sqrt());
            let att = masked_softmax(&att);
            let out = matmul(&att, &vh);
            heads.push(out);
        }
        let concat = concat_cols(&heads);
        self.wo.forward(&concat)
    }
    pub fn params(&self) -> Vec<T> {
        let mut p = self.wq.params();
        p.extend(self.wk.params());
        p.extend(self.wv.params());
        p.extend(self.wo.params());
        p
    }
}

pub struct FFN { pub w1: Linear, pub w2: Linear }

impl FFN {
    pub fn new(c: usize, seed: u64) -> Self {
        Self { w1: Linear::new(c, 4*c, seed+5), w2: Linear::new(4*c, c, seed+6) }
    }
    pub fn forward(&self, x: &T) -> T {
        let h = self.w1.forward(x);
        let h = gelu(&h);
        self.w2.forward(&h)
    }
    pub fn params(&self) -> Vec<T> {
        let mut p = self.w1.params();
        p.extend(self.w2.params());
        p
    }
}

pub struct Block {
    pub ln1_g: T, pub ln1_b: T,
    pub att: Attention,
    pub ln2_g: T, pub ln2_b: T,
    pub ffn: FFN,
}

impl Block {
    pub fn new(c: usize, n_head: usize, seed: u64) -> Self {
        Self {
            ln1_g: leaf(vec![1.0; c], vec![c]),
            ln1_b: leaf(vec![0.0; c], vec![c]),
            att: Attention::new(c, n_head, seed),
            ln2_g: leaf(vec![1.0; c], vec![c]),
            ln2_b: leaf(vec![0.0; c], vec![c]),
            ffn: FFN::new(c, seed),
        }
    }
    pub fn forward(&self, x: &T, pos_offset: usize) -> T {
        let h = layernorm(x, &self.ln1_g, &self.ln1_b, 1e-5);
        let h = self.att.forward(&h, pos_offset);
        let x = add(x, &h);
        let h = layernorm(&x, &self.ln2_g, &self.ln2_b, 1e-5);
        let h = self.ffn.forward(&h);
        add(&x, &h)
    }
    pub fn params(&self) -> Vec<T> {
        let mut p = vec![self.ln1_g.clone(), self.ln1_b.clone()];
        p.extend(self.att.params());
        p.push(self.ln2_g.clone());
        p.push(self.ln2_b.clone());
        p.extend(self.ffn.params());
        p
    }
}
