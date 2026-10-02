// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// AI GENERATED
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

//! notes: semantic search over a folder of markdown notes.
//!
//!   notes index            embed every section of every .md file under $NOTES_DIR
//!   notes <query words>    show the sections closest in meaning to the query
//!
//! $NOTES_DIR defaults to ~/notes. The index is a plain text file saved as
//! $NOTES_DIR/.notes-index.tsv (hidden, so rg skips it).

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::path::{Path, PathBuf};
use tokenizers::{Tokenizer, TruncationParams};

// The model is compiled into the binary, so it works with no downloads.
// https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
const MODEL_WEIGHTS: &[u8] = include_bytes!("../model/model.safetensors");
const MODEL_CONFIG: &str = include_str!("../model/config.json");
const MODEL_TOKENIZER: &[u8] = include_bytes!("../model/tokenizer.json");
const MAX_TOKENS: usize = 256; // all-MiniLM-L6-v2 was trained on inputs up to this length
const RESULTS: usize = 5;

/// One heading's worth of text from a notes file.
struct Section {
    file: String,
    line: usize,
    heading: String, // full heading path, e.g. "Vim > vim edit many files"
    text: String,
}

/// A section plus its embedding, as stored in the index.
struct Entry {
    file: String,
    line: usize,
    heading: String,
    vector: Vec<f32>,
}

fn main() {
    let notes_dir = notes_dir();
    let index_file = notes_dir.join(".notes-index.tsv");
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        eprintln!("usage: notes index | notes <query>");
        std::process::exit(1);
    }

    let embedder = Embedder::new();

    if args[0] == "index" {
        build_index(&embedder, &notes_dir, &index_file);
    } else {
        search(&embedder, &index_file, &args.join(" "));
    }
}

// Defaults to ~/notes
fn notes_dir() -> PathBuf {
    match std::env::var("NOTES_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap()).join("notes"),
    }
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

fn build_index(embedder: &Embedder, notes_dir: &Path, index_file: &Path) {
    let mut files = Vec::new();
    find_markdown_files(notes_dir, &mut files);

    let mut out = String::new();
    let mut count = 0;
    for path in &files {
        let content = std::fs::read_to_string(path).unwrap_or_default();
        let relative = path.strip_prefix(notes_dir).unwrap().display().to_string();

        for section in split_into_sections(&relative, &content) {
            // The file name and heading path give the model useful context.
            let input = format!("{} > {}\n{}", section.file, section.heading, section.text);
            let vector = embedder.embed(&input);

            let numbers: Vec<String> = vector.iter().map(|x| x.to_string()).collect();
            out += &format!(
                "{}\t{}\t{}\t{}\n",
                section.file,
                section.line,
                section.heading,
                numbers.join(",")
            );
            count += 1;
        }
    }

    std::fs::write(index_file, out).expect("could not write index");
    println!("indexed {count} sections from {} files", files.len());
}

fn find_markdown_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("could not read notes dir") {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy();
        if name.starts_with('.') {
            continue; // skip .git, the index itself, etc.
        }
        if path.is_dir() {
            find_markdown_files(&path, files);
        } else if name.ends_with(".md") {
            files.push(path);
        }
    }
}

/// Splits a markdown file at every heading. Each section remembers the
/// headings above it, so "## Many files" under "# Vim" becomes "Vim > Many files".
fn split_into_sections(file: &str, content: &str) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut heading_stack: Vec<String> = Vec::new();
    let mut current = Section {
        file: file.to_string(),
        line: 1,
        heading: String::new(),
        text: String::new(),
    };
    let mut in_code_block = false;

    for (i, line) in content.lines().enumerate() {
        if line.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
        }

        // A "# comment" inside a code block is not a heading.
        let is_heading = !in_code_block && line.starts_with('#');
        if !is_heading {
            current.text += line;
            current.text += "\n";
            continue;
        }

        // Save the section we were building (headings with no text of their
        // own are skipped), then start a new one.
        if !current.text.trim().is_empty() {
            sections.push(current);
        }

        let level = line.chars().take_while(|&c| c == '#').count();
        let title = line.trim_start_matches('#').trim().to_string();
        heading_stack.truncate(level - 1);
        heading_stack.push(title);

        current = Section {
            file: file.to_string(),
            line: i + 1,
            heading: heading_stack.join(" > "),
            text: String::new(),
        };
    }
    if !current.text.trim().is_empty() {
        sections.push(current);
    }
    sections
}

// ---------------------------------------------------------------------------
// Searching
// ---------------------------------------------------------------------------

fn search(embedder: &Embedder, index_file: &Path, query: &str) {
    let entries = load_index(index_file);
    let query_vector = embedder.embed(query);

    // Vectors are normalized, so the dot product is the cosine similarity.
    let mut scored: Vec<(f32, &Entry)> = entries
        .iter()
        .map(|e| (dot(&query_vector, &e.vector), e))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));

    for (score, entry) in scored.iter().take(RESULTS) {
        println!("{score:.3}  {}:{}  {}", entry.file, entry.line, entry.heading);
    }
}

fn load_index(index_file: &Path) -> Vec<Entry> {
    let text = std::fs::read_to_string(index_file).expect("no index found, run `notes index` first");
    text.lines()
        .map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            Entry {
                file: parts[0].to_string(),
                line: parts[1].parse().unwrap(),
                heading: parts[2].to_string(),
                vector: parts[3].split(',').map(|x| x.parse().unwrap()).collect(),
            }
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Embedding model (all-MiniLM-L6-v2, a small BERT that outputs 384 numbers)
// ---------------------------------------------------------------------------

struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
}

impl Embedder {
    fn new() -> Self {
        let config: Config = serde_json::from_str(MODEL_CONFIG).unwrap();
        let weights = VarBuilder::from_slice_safetensors(MODEL_WEIGHTS, DType::F32, &Device::Cpu).unwrap();
        let model = BertModel::load(weights, &config).unwrap();
        let mut tokenizer = Tokenizer::from_bytes(MODEL_TOKENIZER).unwrap();
        // tokenizer.json pads everything to 128 tokens by default. Those padding
        // tokens would get averaged in and drown out the real text, so turn it off.
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams { max_length: MAX_TOKENS, ..Default::default() }))
            .unwrap();
        Embedder { model, tokenizer }
    }

    /// Turns text into a 384-number vector of length 1. Texts with similar
    /// meaning end up with vectors pointing in similar directions.
    fn embed(&self, text: &str) -> Vec<f32> {
        let encoding = self.tokenizer.encode(text, true).unwrap();
        let input_ids = Tensor::new(encoding.get_ids(), &Device::Cpu).unwrap().unsqueeze(0).unwrap();
        let token_type_ids = input_ids.zeros_like().unwrap();

        // One output vector per token: shape [1, tokens, 384]
        let output = self.model.forward(&input_ids, &token_type_ids, None).unwrap();

        // Average the token vectors into one vector, then scale it to length 1.
        let mean = output.mean(1).unwrap().squeeze(0).unwrap();
        let vector: Vec<f32> = mean.to_vec1().unwrap();
        let length = dot(&vector, &vector).sqrt();
        vector.iter().map(|x| x / length).collect()
    }
}
