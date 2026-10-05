use llm_rs::bpe_tokenizer::BpeTokenizer;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();

    let data_path = root.join("data").join("corpus_gutenberg.txt");
    let out_path  = root.join("models").join("tokenizer_bpe.json");

    if args.get(1).map(|s| s.as_str()) == Some("test") {
        let tok = BpeTokenizer::load(out_path.to_str().unwrap())
            .expect("Impossible de charger le tokenizer");
        for s in ["To be or not to be", "the quick brown fox", "hello world"] {
            let ids = tok.encode(s);
            let back = tok.decode(&ids);
            let status = if back == s { "OK" } else { "FAIL" };
            println!("{:>25} -> {:3} tokens -> {:?}  {}", s, ids.len(), back, status);
        }
        return;
    }

    println!("Lecture de {}", data_path.display());
    let text = fs::read_to_string(&data_path).expect("corpus introuvable");
    let sample = &text[..text.len().min(1_000_000)];

    let mut tok = BpeTokenizer::new(1024);
    tok.train(sample, true);

    fs::create_dir_all(root.join("models")).unwrap();
    tok.save(out_path.to_str().unwrap()).unwrap();
    println!("Tokenizer sauvegarde dans {}", out_path.display());

    let raw = text.len().min(500_000);
    let ids = tok.encode(&text[..raw]);
    println!("500 KB -> {} tokens  |  ratio {:.2} octets/token",
        ids.len(), raw as f64 / ids.len() as f64);
}
