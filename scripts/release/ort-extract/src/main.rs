use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::path::PathBuf;

use lzma_rust2::Lzma2Reader;

fn main() {
    if let Err(error) = extract() {
        eprintln!("ORT extraction failed: {error}");
        std::process::exit(1);
    }
}

fn extract() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let input = PathBuf::from(arguments.next().ok_or("missing input archive")?);
    let output = PathBuf::from(arguments.next().ok_or("missing output directory")?);
    if arguments.next().is_some() {
        return Err("expected exactly: noperson-ort-extract <archive> <output-directory>".into());
    }

    fs::create_dir_all(&output)?;
    let archive = File::open(&input)?;
    let decoder = Lzma2Reader::new(archive, 1 << 26, None);
    tar::Archive::new(decoder).unpack(&output)?;
    Ok(())
}
