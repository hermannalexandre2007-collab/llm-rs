use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone)]
pub struct CharTokenizer {
    pub chars: Vec<char>,
    pub stoi: HashMap<char, usize>,
    pub itos: HashMap<usize, char>,
}

impl CharTokenizer {
    pub fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let stoi = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let itos = chars.iter().enumerate().map(|(i, &c)| (i, c)).collect();

        Self { chars, stoi, itos }
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    pub fn encode(&self, s: &str) -> Vec<usize> {
        s.chars()
            .filter_map(|c| self.stoi.get(&c).copied())
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|i| self.itos.get(i).copied())
            .collect()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let s: String = self.chars.iter().collect();
        fs::write(path, s)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let s = fs::read_to_string(path)?;
        Ok(Self::new(&s))
    }
}
