// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// AI GENERATED
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# notes

Semantic search over a folder of markdown notes. The embedding model
(all-MiniLM-L6-v2) is compiled into the binary, so there's nothing to install
or download at runtime.

## Build

Download the model files into `model/` (they get baked in at compile time):

```bash
mkdir -p model && cd model
for f in model.safetensors config.json tokenizer.json; do
  curl -LO https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f
done
cd ..
cargo build --release
cp target/release/notes ~/.local/bin/   # or anywhere on your PATH
```

The binary is about 100 MB, almost all of it model weights.

## Use

```bash
export NOTES_DIR=~/notes      # defaults to ~/notes
notes index                   # embed every section; re-run after editing notes
notes vim multiple files      # search
```

Output is `score  file:line  heading path`, best match first:

```
0.625  linux_tools/linux.md:5  Linux tools > vim > vim edit many files
0.416  linux_tools/linux.md:9  Linux tools > vim > vim macros
```

## How it works

1. Each `.md` file is split at its headings. Every section is embedded together
   with its file name and heading path (`linux.md > Vim > vim edit many files`).
2. Embeddings are 384 floats each, saved in `$NOTES_DIR/.notes-index.tsv`.
3. A search embeds the query and ranks every section by cosine similarity.
