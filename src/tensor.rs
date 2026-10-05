use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

pub type T = Rc<RefCell<Tensor>>;

pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub grad: Vec<f32>,
    pub parents: Vec<T>,
    pub backward: Option<Box<dyn Fn(&[f32])>>,
}

pub fn leaf(data: Vec<f32>, shape: Vec<usize>) -> T {
    let n = data.len();
    Rc::new(RefCell::new(Tensor { data, shape, grad: vec![0.0; n], parents: vec![], backward: None }))
}

pub fn rand_init(n: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 40) as u32) as f32 / ((1u32 << 24) as f32);
        (u * 2.0 - 1.0) * scale
    }).collect()
}

fn make(data: Vec<f32>, shape: Vec<usize>, parents: Vec<T>) -> T {
    let n = data.len();
    Rc::new(RefCell::new(Tensor { data, shape, grad: vec![0.0; n], parents, backward: None }))
}

pub fn backward(root: &T) {
    let mut topo: Vec<T> = vec![];
    let mut visited: HashSet<usize> = HashSet::new();
    fn build(n: &T, v: &mut HashSet<usize>, t: &mut Vec<T>) {
        let key = Rc::as_ptr(n) as usize;
        if !v.insert(key) { return; }
        let parents = n.borrow().parents.clone();
        for p in &parents { build(p, v, t); }
        t.push(n.clone());
    }
    build(root, &mut visited, &mut topo);
    // NOTE: on n'efface PAS les gradients ici. On accumule.
    // L'appelant doit zero_grad() explicitement.
    { let mut r = root.borrow_mut(); for g in r.grad.iter_mut() { *g += 1.0; } }
    for n in topo.iter().rev() {
        let grad = n.borrow().grad.clone();
        let bw = n.borrow_mut().backward.take();
        if let Some(bw) = bw { bw(&grad); }
    }
}

pub fn add(a: &T, b: &T) -> T {
    let (d, s) = {
        let ab = a.borrow(); let bb = b.borrow();
        (ab.data.iter().zip(&bb.data).map(|(x,y)| x+y).collect::<Vec<_>>(), ab.shape.clone())
    };
    let ac = a.clone(); let bc = b.clone();
    let o = make(d, s, vec![a.clone(), b.clone()]);
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut ab = ac.borrow_mut(); let mut bb = bc.borrow_mut();
        for i in 0..g.len() { ab.grad[i] += g[i]; bb.grad[i] += g[i]; }
    }));
    o
}

pub fn matmul(a: &T, b: &T) -> T {
    let (ad, bd, m, n, p) = {
        let ab = a.borrow(); let bb = b.borrow();
        (ab.data.clone(), bb.data.clone(), ab.shape[0], ab.shape[1], bb.shape[1])
    };
    let mut d = vec![0.0; m*p];
    for i in 0..m { for j in 0..p {
        let mut s = 0.0;
        for k in 0..n { s += ad[i*n+k] * bd[k*p+j]; }
        d[i*p+j] = s;
    }}
    let ac = a.clone(); let bc = b.clone();
    let o = make(d, vec![m, p], vec![a.clone(), b.clone()]);
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut ab = ac.borrow_mut(); let mut bb = bc.borrow_mut();
        for i in 0..m { for k in 0..n {
            let mut s = 0.0;
            for j in 0..p { s += g[i*p+j] * bd[k*p+j]; }
            ab.grad[i*n+k] += s;
        }}
        for k in 0..n { for j in 0..p {
            let mut s = 0.0;
            for i in 0..m { s += ad[i*n+k] * g[i*p+j]; }
            bb.grad[k*p+j] += s;
        }}
    }));
    o
}

pub fn transpose(a: &T) -> T {
    let (ad, m, n) = { let ab = a.borrow(); (ab.data.clone(), ab.shape[0], ab.shape[1]) };
    let mut d = vec![0.0; n*m];
    for i in 0..m { for j in 0..n { d[j*m+i] = ad[i*n+j]; } }
    let ac = a.clone();
    let o = make(d, vec![n, m], vec![a.clone()]);
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut ab = ac.borrow_mut();
        for i in 0..m { for j in 0..n { ab.grad[i*n+j] += g[j*m+i]; } }
    }));
    o
}

pub fn add_row(a: &T, b: &T) -> T {
    let (ad, bd, t, c) = {
        let ab = a.borrow(); let bb = b.borrow();
        (ab.data.clone(), bb.data.clone(), ab.shape[0], ab.shape[1])
    };
    let mut d = vec![0.0; t*c];
    for i in 0..t { for j in 0..c { d[i*c+j] = ad[i*c+j] + bd[j]; } }
    let ac = a.clone(); let bc = b.clone();
    let o = make(d, vec![t, c], vec![a.clone(), b.clone()]);
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut ab = ac.borrow_mut(); let mut bb = bc.borrow_mut();
        for i in 0..t { for j in 0..c {
            ab.grad[i*c+j] += g[i*c+j];
            bb.grad[j] += g[i*c+j];
        }}
    }));
    o
}

pub fn scale(a: &T, s: f32) -> T {
    let (d, sh) = {
        let ab = a.borrow();
        (ab.data.iter().map(|x| x*s).collect::<Vec<_>>(), ab.shape.clone())
    };
    let ac = a.clone();
    let o = make(d, sh, vec![a.clone()]);
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut ab = ac.borrow_mut();
        for i in 0..g.len() { ab.grad[i] += g[i] * s; }
    }));
    o
}

pub fn masked_softmax(a: &T) -> T {
    let (ad, t) = { let ab = a.borrow(); (ab.data.clone(), ab.shape[0]) };
    let mut d = vec![0.0; t*t];
    for i in 0..t {
        let mut max = f32::NEG_INFINITY;
        for j in 0..=i { if ad[i*t+j] > max { max = ad[i*t+j]; } }
        let mut sum = 0.0;
        for j in 0..=i { let e = (ad[i*t+j] - max).exp(); d[i*t+j] = e; sum += e; }
        for j in 0..=i { d[i*t+j] /= sum; }
    }
    let ac = a.clone();
    let d_c = d.clone();
    let o = make(d, vec![t, t], vec![a.clone()]);
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut ab = ac.borrow_mut();
        for i in 0..t {
            let mut s = 0.0;
            for j in 0..=i { s += g[i*t+j] * d_c[i*t+j]; }
            for j in 0..=i { ab.grad[i*t+j] += d_c[i*t+j] * (g[i*t+j] - s); }
        }
    }));
    o
}

pub fn layernorm(a: &T, gamma: &T, beta: &T, eps: f32) -> T {
    let (ad, t, c) = { let ab = a.borrow(); (ab.data.clone(), ab.shape[0], ab.shape[1]) };
    let (gd, bd) = { let gb = gamma.borrow(); let bb = beta.borrow(); (gb.data.clone(), bb.data.clone()) };
    let mut mean = vec![0.0; t]; let mut var = vec![0.0; t];
    for i in 0..t {
        let mut s = 0.0;
        for j in 0..c { s += ad[i*c+j]; }
        mean[i] = s / c as f32;
        let mut v = 0.0;
        for j in 0..c { let d = ad[i*c+j] - mean[i]; v += d*d; }
        var[i] = v / c as f32;
    }
    let mut xhat = vec![0.0; t*c]; let mut inv_std = vec![0.0; t];
    for i in 0..t {
        inv_std[i] = 1.0 / (var[i] + eps).sqrt();
        for j in 0..c { xhat[i*c+j] = (ad[i*c+j] - mean[i]) * inv_std[i]; }
    }
    let mut d = vec![0.0; t*c];
    for i in 0..t { for j in 0..c { d[i*c+j] = xhat[i*c+j] * gd[j] + bd[j]; } }
    let ac = a.clone(); let gc = gamma.clone(); let bc = beta.clone();
    let xhat_c = xhat.clone(); let inv_std_c = inv_std.clone();
    let o = make(d, vec![t, c], vec![a.clone(), gamma.clone(), beta.clone()]);
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut ab = ac.borrow_mut();
        let mut gb = gc.borrow_mut();
        let mut bb = bc.borrow_mut();
        let mut dxhat = vec![0.0; t*c];
        for i in 0..t { for j in 0..c {
            dxhat[i*c+j] = g[i*c+j] * gb.data[j];
            gb.grad[j] += g[i*c+j] * xhat_c[i*c+j];
            bb.grad[j] += g[i*c+j];
        }}
        for i in 0..t {
            let inv = inv_std_c[i];
            let mut s1 = 0.0; let mut s2 = 0.0;
            for j in 0..c { s1 += dxhat[i*c+j]; s2 += dxhat[i*c+j] * xhat_c[i*c+j]; }
            for j in 0..c {
                ab.grad[i*c+j] += inv / c as f32 * (c as f32 * dxhat[i*c+j] - s1 - xhat_c[i*c+j] * s2);
            }
        }
    }));
    o
}

pub fn gelu(a: &T) -> T {
    let (ad, sh) = { let ab = a.borrow(); (ab.data.clone(), ab.shape.clone()) };
    let sqrt2pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let mut d = vec![0.0; ad.len()];
    let mut tanh_vals = vec![0.0; ad.len()];
    for i in 0..ad.len() {
        let x = ad[i];
        let inner = sqrt2pi * (x + 0.044715 * x*x*x);
        let th = inner.tanh();
        tanh_vals[i] = th;
        d[i] = 0.5 * x * (1.0 + th);
    }
    let ac = a.clone();
    let o = make(d, sh, vec![a.clone()]);
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut ab = ac.borrow_mut();
        for i in 0..g.len() {
            let x = ad[i]; let th = tanh_vals[i];
            let dg = 0.5*(1.0+th) + 0.5*x*(1.0-th*th)*sqrt2pi*(1.0 + 3.0*0.044715*x*x);
            ab.grad[i] += g[i] * dg;
        }
    }));
    o
}

pub fn cross_entropy(logits: &T, targets: &[usize]) -> T {
    let (ld, t, v) = { let lb = logits.borrow(); (lb.data.clone(), lb.shape[0], lb.shape[1]) };
    let mut probs = vec![0.0; t*v];
    let mut loss = 0.0;
    for i in 0..t {
        let mut max = f32::NEG_INFINITY;
        for j in 0..v { if ld[i*v+j] > max { max = ld[i*v+j]; } }
        let mut sum = 0.0;
        for j in 0..v { let e = (ld[i*v+j] - max).exp(); probs[i*v+j] = e; sum += e; }
        for j in 0..v { probs[i*v+j] /= sum; }
        loss -= probs[i*v + targets[i]].ln();
    }
    loss /= t as f32;
    let lc = logits.clone();
    let tgt_c = targets.to_vec();
    let probs_c = probs.clone();
    let o = make(vec![loss], vec![1], vec![logits.clone()]);
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut lb = lc.borrow_mut();
        let sc = g[0] / t as f32;
        for i in 0..t { for j in 0..v {
            let mut gr = probs_c[i*v+j];
            if j == tgt_c[i] { gr -= 1.0; }
            lb.grad[i*v+j] += gr * sc;
        }}
    }));
    o
}

pub fn embedding_lookup(w: &T, idx: &[usize]) -> T {
    let (wd, c) = { let wb = w.borrow(); (wb.data.clone(), wb.shape[1]) };
    let t = idx.len();
    let mut d = vec![0.0; t*c];
    for i in 0..t {
        let r = idx[i];
        for j in 0..c { d[i*c+j] = wd[r*c+j]; }
    }
    let wc = w.clone();
    let idx_c = idx.to_vec();
    let o = make(d, vec![t, c], vec![w.clone()]);
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut wb = wc.borrow_mut();
        for i in 0..t {
            let r = idx_c[i];
            for j in 0..c { wb.grad[r*c+j] += g[i*c+j]; }
        }
    }));
    o
}

pub fn zero_grad(params: &[T]) {
    for p in params {
        let mut pb = p.borrow_mut();
        for g in pb.grad.iter_mut() { *g = 0.0; }
    }
}

pub fn slice_cols(a: &T, start: usize, len: usize) -> T {
    let (ad, t, c) = { let ab = a.borrow(); (ab.data.clone(), ab.shape[0], ab.shape[1]) };
    let mut d = vec![0.0; t * len];
    for i in 0..t {
        for j in 0..len {
            d[i*len + j] = ad[i*c + start + j];
        }
    }
    let ac = a.clone();
    let o = make(d, vec![t, len], vec![a.clone()]);
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut ab = ac.borrow_mut();
        for i in 0..t {
            for j in 0..len {
                ab.grad[i*c + start + j] += g[i*len + j];
            }
        }
    }));
    o
}

pub fn concat_cols(parts: &[T]) -> T {
    let t = parts[0].borrow().shape[0];
    let total_c: usize = parts.iter().map(|p| p.borrow().shape[1]).sum();
    let mut d = vec![0.0; t * total_c];
    let mut off = 0;
    for p in parts {
        let pb = p.borrow();
        let c = pb.shape[1];
        for i in 0..t {
            for j in 0..c {
                d[i*total_c + off + j] = pb.data[i*c + j];
            }
        }
        off += c;
    }
    let parts_c: Vec<T> = parts.to_vec();
    let o = make(d, vec![t, total_c], parts_c.clone());
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut off = 0;
        for p in &parts_c {
            let mut pb = p.borrow_mut();
            let c = pb.shape[1];
            for i in 0..t {
                for j in 0..c {
                    pb.grad[i*c + j] += g[i*total_c + off + j];
                }
            }
            off += c;
        }
    }));
    o
}


pub fn rope_apply(x: &T, head_dim: usize, base: f32, pos_offset: usize) -> T {
    let (xd, t, c) = {
        let xb = x.borrow();
        (xb.data.clone(), xb.shape[0], xb.shape[1])
    };
    let n_head = c / head_dim;
    let mut out = vec![0.0f32; t * c];
    let mut cos_tab = vec![0.0f32; t * (head_dim / 2)];
    let mut sin_tab = vec![0.0f32; t * (head_dim / 2)];
    for i in 0..t {
        let pos = (pos_offset + i) as f32;
        for j in 0..head_dim / 2 {
            let theta = pos / base.powf(2.0 * (j as f32) / (head_dim as f32));
            let (s, co) = theta.sin_cos();
            cos_tab[i * (head_dim/2) + j] = co;
            sin_tab[i * (head_dim/2) + j] = s;
        }
    }
    for i in 0..t {
        for h in 0..n_head {
            let base_idx = h * head_dim;
            for j in 0..head_dim / 2 {
                let co = cos_tab[i * (head_dim/2) + j];
                let s  = sin_tab[i * (head_dim/2) + j];
                let a = xd[i*c + base_idx + 2*j];
                let b = xd[i*c + base_idx + 2*j + 1];
                out[i*c + base_idx + 2*j]     = a * co - b * s;
                out[i*c + base_idx + 2*j + 1] = a * s  + b * co;
            }
        }
    }
    let xc = x.clone();
    let cos_c = cos_tab.clone();
    let sin_c = sin_tab.clone();
    let o = make(out, vec![t, c], vec![x.clone()]);
    o.borrow_mut().backward = Some(Box::new(move |g: &[f32]| {
        let mut xb = xc.borrow_mut();
        for i in 0..t {
            for h in 0..n_head {
                let base_idx = h * head_dim;
                for j in 0..head_dim / 2 {
                    let co = cos_c[i * (head_dim/2) + j];
                    let s  = sin_c[i * (head_dim/2) + j];
                    let ga = g[i*c + base_idx + 2*j];
                    let gb = g[i*c + base_idx + 2*j + 1];
                    xb.grad[i*c + base_idx + 2*j]     +=  ga * co + gb * s;
                    xb.grad[i*c + base_idx + 2*j + 1] += -ga * s  + gb * co;
                }
            }
        }
    }));
    o
}
