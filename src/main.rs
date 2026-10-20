//! notes: semantic search over a folder of markdown notes.
//!
//!   notes index            embed every section of every .md file under $NOTES_DIR
//!   notes <query words>    show the sections closest in meaning to the query
//!
//! $NOTES_DIR defaults to ~/notes. The index is a plain text file saved as
//! $NOTES_DIR/.notes-index.tsv (hidden, so rg skips it).

mod tui;
use std::io::{self, Read, Write};
use std::fs::read_to_string;
use candle_core::{DType, Device, Tensor};
use std::process::{Command, Stdio};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::path::{Path, PathBuf};
use tokenizers::{Tokenizer, TruncationParams};
use anyhow::Result;
use data_encoding::HEXUPPER;
use ring::digest::{Context, Digest, SHA256};
use std::fs::File;
use std::io::{BufReader};

// The model is compiled into the binary, so it works with no downloads.
// https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
const MODEL_WEIGHTS: &[u8] = include_bytes!("../model/model.safetensors");
const MODEL_CONFIG: &str = include_str!("../model/config.json");
const MODEL_TOKENIZER: &[u8] = include_bytes!("../model/tokenizer.json");
const MAX_TOKENS: usize = 256; // all-MiniLM-L6-v2 was trained on inputs up to this length
const RESULTS: usize = 10;

// https://rust-lang-nursery.github.io/rust-cookbook/cryptography/hashing.html
fn sha256_digest<R: Read>(mut reader: R) -> Result<Digest> {
    let mut context = Context::new(&SHA256);
    let mut buffer = [0; 1024];

    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        context.update(&buffer[..count]);
    }

    Ok(context.finish())
}


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
const RAW_FLAGS: &[&str] = &[
    "-icanon", "-echo", "-isig", "-opost", "min", "1", "time", "0",
];

fn stty(args: &[&str]) -> io::Result<String> {
    let out = Command::new("stty")
        .args(args)
        .stdin(Stdio::inherit())
        .stderr(Stdio::inherit())
        .output()?;
    if !out.status.success() {
        return Err(io::Error::new(io::ErrorKind::Other, "stty failed"));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

struct RawMode {
    saved: String,
}

impl RawMode {
    fn enable() -> io::Result<RawMode> {
        let saved = stty(&["-g"])?;
        stty(RAW_FLAGS)?;
        Ok(RawMode { saved })
    }
    fn restore(&self) {
        let args: Vec<&str> = self.saved.split_whitespace().collect();
        if !args.is_empty() {
            let _ = stty(&args);
        }
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        self.restore();
        // show cursor, reset attrs, leave the prompt on a fresh line
        print!("\x1b[0m\x1b[?25h\r\n");
        let _ = io::stdout().flush();
    }
}

fn main() {
    let mut stdin = io::stdin();
    let mut buf = [0u8; 8];
    let notes_dir = notes_dir();
    let index_folder = notes_dir.join(".index");
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        eprintln!("usage: notes index | notes <query>");
        std::process::exit(1);
    }

    let embedder = Embedder::new();
    build_index(&embedder, &notes_dir, &index_folder);

    let _raw = match RawMode::enable() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("audit-ui: could not enter raw mode: {}", e);
            return;
        }
    };
    print!("\x1b[?25l"); // hide cursor
    let _ = io::stdout().flush();


    if args[0] == "index" {
    } else {
        let search_str = &args.join(" ");
        let results = search(&embedder, &index_folder, search_str);
        let result_strings = result_strings(&results);

        let mut selected_entry = 0;
        loop {
            tui::render(selected_entry, &result_strings,search_str);
            let n = match stdin.read(&mut buf) {
                Ok(0) => break, // EOF
                Ok(n) => n,
                Err(_) => break,
            };
            match parse_key(&buf[..n]) {
                Action::Up =>  {if selected_entry > 0 {selected_entry -= 1;} },
                Action::Down  =>  {if selected_entry < result_strings.len()-1 {selected_entry += 1;} },
                Action::Go =>  {
                    let target_path = format!("{}/{}",notes_dir.to_str().expect("unable to read notes dir"),&results[selected_entry].1.file);
                    let line_num = format!("+{}",&results[selected_entry].1.line);
                    let _ = Command::new("vim").arg(target_path).arg(line_num.to_string()).status();
                    break;
                },
                Action::Quit => break,
                Action::None => {},
            }
            
        }
    }
}

enum Action {
    Up,
    Down,
    Go,
    Quit,
    None,
}

fn parse_key(key: &[u8]) -> Action {
    if key.len() >= 3 && key[0] == 0x1b && key[1] == b'[' {
        return match key[2] {
            b'A' => Action::Up,
            b'B' => Action::Down,
            _ => Action::None,
        };
    }
    match key[0] {
        b'l' => Action::Up,
        b'k' => Action::Down,
        b'\r' | b'\n' => Action::Go,
        b'q' | 0x03 => Action::Quit,
        _ => Action::None,
    }
}

// Defaults to ~/notes
// AI GENERATED
fn notes_dir() -> PathBuf {
    match std::env::var("NOTES_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap()).join("notes"),
    }
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

fn build_index(embedder: &Embedder, notes_dir: &Path, index_folder: &Path) {
    let mut files = Vec::new();
    find_markdown_files(notes_dir, &mut files);

    let mut section_count = 0;
    let mut file_count = 0;
    for path in &files {
        let mut out = String::new();

        let mut context = Context::new(&SHA256);
        context.update(path.clone().into_os_string().into_string().expect("issue with path").as_str().as_bytes());
        let hash = context.finish();
        let hash_hex = HEXUPPER.encode(hash.as_ref());
        let index_file = index_folder.join(hash_hex);
        if let Some(old_hash) = read_to_string(&index_file).expect("unable to read index file for hash").lines().next() {
            if let Ok(file_contents) = File::open(path) {
                let reader = BufReader::new(file_contents);
                if let Ok(digest) = sha256_digest(reader) {
                    let content_hash = HEXUPPER.encode(digest.as_ref());
                    if content_hash == old_hash {continue;}
                    println!("{} != {}",content_hash, old_hash);

                    out += &format!("{}\n", content_hash);
                }
            }
        }
        println!("{}",path.display());


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
            section_count += 1;
        }
        file_count += 1;
        std::fs::write(index_file, out).expect("could not write index");
    }

    println!("indexed {} sections from {} files out of {} files", section_count, file_count, files.len());
}

// AI GENERATED
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
// AI GENERATED
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

// AI GENERATED
fn search(embedder: &Embedder, index_folder: &Path, query: &str) -> Vec<(f32, Entry)>{
    let entries = load_index(index_folder);
    let query_vector = embedder.embed(query);

    // Vectors are normalized, so the dot product is the cosine similarity.
    let mut scored: Vec<(f32, Entry)> = entries
        .into_iter()
        .map(|e| (dot(&query_vector, &e.vector), e))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.truncate(RESULTS);
    scored
}

fn result_strings(scored_entries: &Vec<(f32, Entry)>) -> Vec<String> {
    let mut out : Vec<String> = Vec::new();
    for (score, entry) in scored_entries.iter() {
        let entry_string = format!("{:.3}  {}:{}  {}",score, entry.file, entry.line, entry.heading );
        out.push(entry_string);
    }
    out
}

fn load_index(index_folder: &Path) -> Vec<Entry> {
    let files = std::fs::read_dir(index_folder).unwrap();
    let mut out:Vec<Entry> = vec![];

    for file in files {
        out.append(&mut load_index_file(&file.unwrap().path()));
    }
    out
}

// AI GENERATED
fn load_index_file(index_folder: &Path) -> Vec<Entry> {
    
    let text = std::fs::read_to_string(index_folder).expect("no index found, run `notes index` first");
    text.lines().skip(1)
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

// AI GENERATED
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
    // AI GENERATED
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
    // AI GENERATED
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
