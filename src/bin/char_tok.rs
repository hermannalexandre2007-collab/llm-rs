use llm_rs::char_tokenizer::CharTokenizer;

fn main() {
    let sample = "Bonjour le monde ! Comment ca va ?";
    let tok = CharTokenizer::new(sample);

    println!("Vocab size : {}", tok.vocab_size());
    println!("Vocab      : {:?}", tok.chars);

    let ids = tok.encode(sample);
    println!("Encode     : {:?}", ids);

    let back = tok.decode(&ids);
    println!("Decode     : {}", back);

    assert_eq!(back, sample, "Round-trip casse !");
    println!("Round-trip OK");
}
