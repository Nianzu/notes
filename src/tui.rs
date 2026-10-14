use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

// https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
pub trait Colorize {
    fn red(&self) -> String;
    fn yellow(&self) -> String;
    fn green(&self) -> String;
    fn bold(&self) -> String;
    fn grey(&self) -> String;
    fn cyan(&self) -> String;
    fn reset_fg(&self) -> String;
    fn reset(&self) -> String;
    fn reverse_colors(&self) -> String;
}

impl Colorize for str {
    fn red(&self) -> String {
        format!("\x1b[31m{self}")
    }
    fn yellow(&self) -> String {
        format!("\x1b[33m{self}")
    }
    fn green(&self) -> String {
        format!("\x1b[32m{self}")
    }
    fn bold(&self) -> String {
        format!("\x1b[1m{self}")
    }
    fn grey(&self) -> String {
        format!("\x1b[2m{self}")
    }
    fn cyan(&self) -> String {
        format!("\x1b[36m{self}")
    }
    fn reset_fg(&self) -> String {
        format!("\x1b[39m{self}")
    }
    fn reset(&self) -> String {
        format!("{self}\x1b[0m")
    }
    fn reverse_colors(&self) -> String {
        format!("\x1b[7m{self}")
    }
}

pub fn render(
    selected: usize,
    lines: &Vec<String>,
    search_str: &str
) {
    let mut out = String::new();
    out.push_str("\x1b[2J\x1b[H"); // clear + home

    out.push_str(&format!("Top embeddings for \'{search_str}\'").bold().reset());

    out.push_str("\r\n");
    // TODO grab keybinds from actual keybinds
    out.push_str(
        &"[k/l] move  [enter] open/cd  [h] up  \
         [u]nread [p]artial [a]pproved  [space] cycle  [f]lag  [q]uit\r\n"
            .grey()
            .reset(),
    );
    out.push_str("\r\n");

    for (i, line) in lines.iter().enumerate() {
        if i == selected {
            out.push_str(&"".bold());
        }
        out.push_str(&line);
        out.push_str(&"\r\n".reset());
    }

    let stdout = io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(out.as_bytes());
    let _ = lock.flush();
}
