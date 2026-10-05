use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

type Pair = (usize, usize);

#[derive(Serialize, Deserialize)]
struct BpeData {
    vocab_size: usize,
    merges: Vec<(Pair, usize)>,
}

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    pub vocab_size: usize,
    pub merges: HashMap<Pair, usize>,
    pub vocab: HashMap<usize, Vec<u8>>,
}

fn get_stats(ids: &[usize]) -> HashMap<Pair, usize> {
    let mut counts: HashMap<Pair, usize> = HashMap::new();
    for w in ids.windows(2) {
        *counts.entry((w[0], w[1])).or_insert(0) += 1;
    }
    counts
}

fn merge_ids(ids: &[usize], pair: Pair, new_id: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            out.push(new_id);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}

impl BpeTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            merges: HashMap::new(),
            vocab: HashMap::new(),
        }
    }

    pub fn train(&mut self, text: &str, verbose: bool) {
        let mut ids: Vec<usize> = text.bytes().map(|b| b as usize).collect();

        let num_merges = self.vocab_size.saturating_sub(256);

        for i in 0..num_merges {
            let stats = get_stats(&ids);
            if stats.is_empty() {
                break;
            }

            let (&pair, _) = stats.iter().max_by_key(|(_, &c)| c).unwrap();

            let new_id = 256 + i;
            ids = merge_ids(&ids, pair, new_id);
            self.merges.insert(pair, new_id);

            if verbose && i % 50 == 0 {
                println!("merge {}: {:?} -> {}", i, pair, new_id);
            }
        }

        self.vocab.clear();
        for i in 0..256 {
            self.vocab.insert(i, vec![i as u8]);
        }
        let mut merges_sorted: Vec<_> = self.merges.iter().collect();
        merges_sorted.sort_by_key(|(_, &v)| v);
        for (&(p0, p1), &idx) in merges_sorted {
            let a = self.vocab.get(&p0).cloned().unwrap_or_default();
            let b = self.vocab.get(&p1).cloned().unwrap_or_default();
            let mut merged = a;
            merged.extend(b);
            self.vocab.insert(idx, merged);
        }

        println!("Entrainement termine : {} merges", self.merges.len());
        println!("Vocab final : {} tokens", self.vocab.len());
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids: Vec<usize> = text.bytes().map(|b| b as usize).collect();

        while ids.len() >= 2 {
            let stats = get_stats(&ids);
            let pair = stats
                .keys()
                .filter_map(|p| self.merges.get(p).map(|&rank| (*p, rank)))
                .min_by_key(|&(_, rank)| rank)
                .map(|(p, _)| p);

            let Some(pair) = pair else { break };
            let new_id = self.merges[&pair];
            ids = merge_ids(&ids, pair, new_id);
        }

        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(b) = self.vocab.get(id) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut merges: Vec<(Pair, usize)> = self
            .merges
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        merges.sort_by_key(|(_, v)| *v);

        let data = BpeData {
            vocab_size: self.vocab_size,
            merges,
        };
        let json = serde_json::to_string(&data)?;
        fs::write(path, json)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let data: BpeData = serde_json::from_str(&json)?;

        let mut tok = Self::new(data.vocab_size);
        tok.vocab = (0..256).map(|i| (i, vec![i as u8])).collect();

        for (pair, idx) in data.merges {
            tok.merges.insert(pair, idx);
            let a = tok.vocab.get(&pair.0).cloned().unwrap_or_default();
            let b = tok.vocab.get(&pair.1).cloned().unwrap_or_default();
            let mut merged = a;
            merged.extend(b);
            tok.vocab.insert(idx, merged);
        }

        Ok(tok)
    }
}
