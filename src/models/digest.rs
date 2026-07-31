use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const HASH_BUFFER_BYTES: usize = 8 * 1024 * 1024;

pub fn file_blake3(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update_rayon(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}
