# Vundle-
Project 
burn = { version = "0.13", features = ["train"] }
zip = "0.6"
quick-xml = "0.30"
tokenizers = "0.15"
rand = "0.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
use std::fs::File;
use std::io::Read;
use zip::ZipArchive;

pub fn extract_docx(path: &str) -> String {
    let file = File::open(path).unwrap();
    let mut archive = ZipArchive::new(file).unwrap();
    let mut xml = String::new();

    archive.by_name("word/document.xml")
        .unwrap()
        .read_to_string(&mut xml)
        .unwrap();

    xml
}
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
let encoding = tokenizer.encode(
    format!("[CLS] {} [SEP] {} [SEP]", question, context),
    true
).unwrap();

let input_ids = encoding.get_ids().to_vec();
use burn::data::dataset::Dataset;

#[derive(Clone)]
pub struct QASample {
    pub input_ids: Vec<usize>,
    pub start_pos: usize,
    pub end_pos: usize,
}

pub struct QADataset {
    samples: Vec<QASample>,
}

impl Dataset<QASample> for QADataset {
    fn get(&self, index: usize) -> Option<QASample> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}
use burn::data::dataloader::DataLoaderBuilder;

let train_loader = DataLoaderBuilder::new(train_dataset)
    .batch_size(config.batch_size)
    .shuffle(config.shuffle)
    .build();
let split = (samples.len() as f32 * 0.8) as usize;
let train_samples = samples[..split].to_vec();
let val_samples = samples[split..].to_vec();
let split = (samples.len() as f32 * 0.8) as usize;
let train_samples = samples[..split].to_vec();
let val_samples = samples[split..].to_vec();
use burn::nn::{
    Embedding, Linear,
    TransformerEncoder,
    TransformerEncoderConfig
};
use burn::tensor::backend::Backend;
use burn::module::Module;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct QAModel<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    encoder: TransformerEncoder<B>,
    start_head: Linear<B>,
    end_head: Linear<B>,
}
impl<B: Backend> QAModel<B> {
    pub fn new(vocab_size: usize, hidden: usize, max_len: usize) -> Self {

        let encoder_config = TransformerEncoderConfig::new(hidden)
            .with_num_layers(6)
            .with_num_heads(8)
            .with_ffn_hidden(hidden * 4);

        Self {
            token_embedding: Embedding::new(vocab_size, hidden),
            position_embedding: Embedding::new(max_len, hidden),
            encoder: encoder_config.init(),
            start_head: Linear::new(hidden, 1),
            end_head: Linear::new(hidden, 1),
        }
    }
}
pub fn forward(&self, input: Tensor<B, 2>) 
    -> (Tensor<B, 2>, Tensor<B, 2>) {

    let seq_len = input.dims()[1];

    let positions = Tensor::arange(0..seq_len as i64)
        .unsqueeze()
        .expand(input.dims());

    let token_embed = self.token_embedding.forward(input.clone());
    let pos_embed = self.position_embedding.forward(positions);

    let x = token_embed + pos_embed;

    let encoded = self.encoder.forward(x);

    let start_logits = self.start_head.forward(encoded.clone()).squeeze(-1);
    let end_logits = self.end_head.forward(encoded).squeeze(-1);

    (start_logits, end_logits)
}
pub fn forward(&self, input: Tensor<B, 2>) 
    -> (Tensor<B, 2>, Tensor<B, 2>) {

    let seq_len = input.dims()[1];

    let positions = Tensor::arange(0..seq_len as i64)
        .unsqueeze()
        .expand(input.dims());

    let token_embed = self.token_embedding.forward(input.clone());
    let pos_embed = self.position_embedding.forward(positions);

    let x = token_embed + pos_embed;

    let encoded = self.encoder.forward(x);

    let start_logits = self.start_head.forward(encoded.clone()).squeeze(-1);
    let end_logits = self.end_head.forward(encoded).squeeze(-1);

    (start_logits, end_logits)
}
use burn::optim::Adam;

let mut optimizer = Adam::new(config.lr);
for epoch in 0..config.epochs {

    let mut total_loss = 0.0;
    let mut correct = 0;
    let mut total = 0;

    for batch in train_loader.iter() {

        let (start_logits, end_logits) = model.forward(batch.input);

        let loss_start = cross_entropy(start_logits, batch.start_pos);
        let loss_end = cross_entropy(end_logits, batch.end_pos);

        let loss = loss_start + loss_end;

        let grads = loss.backward();

        optimizer.step(&mut model, grads);

        total_loss += loss.to_scalar();
        total += 1;
    }

    println!("Epoch {} Loss {}", epoch, total_loss / total as f32);

    model.save_file(format!("checkpoint_{}.pt", epoch));
}
let model = QAModel::load_file("best_model.pt");
use std::io;

println!("Enter your question:");
let mut question = String::new();
io::stdin().read_line(&mut question).unwrap();
let encoding = tokenizer.encode(
    format!("[CLS] {} [SEP] {} [SEP]", question, context),
    true
).unwrap();

let input_tensor = Tensor::from_vec(encoding.get_ids().to_vec());

let (start_logits, end_logits) = model.forward(input_tensor);

let start = start_logits.argmax(1);
let end = end_logits.argmax(1);

let answer = tokenizer.decode(
    encoding.get_ids()[start..=end].to_vec(),
    true
).unwrap();

println!("Answer: {}", answer);
