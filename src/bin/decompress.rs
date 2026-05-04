use std::env;
use std::fs::File;
use std::io::{Read, Write, BufWriter};

use audio_compressor::entropy::{zigzag_decode, BitReader};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: decompress <fichier.acmp>");
        std::process::exit(1);
    }
    
    let path = &args[1];
    
    let mut file = File::open(path).unwrap_or_else(|e| {
        eprintln!("Erreur lors de l'ouverture de {}: {}", path, e);
        std::process::exit(1);
    });
    
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap_or_else(|e| {
        eprintln!("Erreur de lecture du fichier: {}", e);
        std::process::exit(1);
    });
    
    // Header check
    if buffer.len() < 36 || &buffer[0..4] != b"ACMP" || buffer[4] != 1 {
        eprintln!("Format de fichier invalide ou version non supportée");
        std::process::exit(1);
    }
    
    // Parse ArchiveHeader
    let sample_rate = u32::from_le_bytes(buffer[5..9].try_into().unwrap());
    let channels = u16::from_le_bytes(buffer[9..11].try_into().unwrap());
    let frame_size = u32::from_le_bytes(buffer[11..15].try_into().unwrap());
    let lpc_order = buffer[15] as usize;
    let total_frames = u32::from_le_bytes(buffer[16..20].try_into().unwrap());
    let data_size = u64::from_le_bytes(buffer[20..28].try_into().unwrap());
    let total_samples = u64::from_le_bytes(buffer[28..36].try_into().unwrap());
    
    println!("=== Lecture de l'Archive ===");
    println!("Sample Rate   : {} Hz", sample_rate);
    println!("Channels      : {}", channels);
    println!("Frame Size    : {}", frame_size);
    println!("LPC Order     : {}", lpc_order);
    println!("Total Frames  : {}", total_frames);
    println!("Data Size     : {} octets", data_size);
    println!("Total Samples : {}", total_samples);
    println!();
    
    let payload = &buffer[36..36 + data_size as usize];
    let mut reader = BitReader::new(payload);
    
    let mut reconstructed_samples: Vec<i16> = Vec::with_capacity(total_samples as usize);
    let mut samples_decoded = 0;
    
    for f in 0..total_frames {
        let mut coeffs = vec![0.0; lpc_order];
        for i in 0..lpc_order {
            coeffs[i] = reader.read_f64();
        }
        
        let k = reader.read_bits(5) as u8;
        
        // La dernière frame peut être plus petite
        let remaining = total_samples - samples_decoded;
        let current_frame_len = if remaining < frame_size as u64 {
            remaining as usize
        } else {
            frame_size as usize
        };
        
        let mut frame_samples = Vec::with_capacity(current_frame_len);
        
        for i in 0..current_frame_len {
            let q = reader.read_unary();
            let rem = reader.read_bits(k);
            let z = (q << k) | (rem as u32);
            let residual = zigzag_decode(z);
            
            let mut prediction = 0.0f64;
            let max_k = lpc_order.min(i);
            for j in 0..max_k {
                let sample_prev = frame_samples[i - 1 - j];
                prediction += coeffs[j] * sample_prev as f64;
            }
            
            let pred_i16 = prediction.round() as i16;
            
            // Le résidu a été compressé comme: S - P = R
            // Donc S = R + P
            let sample = (pred_i16 as i32 + residual) as i16;
            frame_samples.push(sample);
            reconstructed_samples.push(sample);
            samples_decoded += 1;
        }
    }
    
    println!("Décodage terminé : {} échantillons reconstruits.", samples_decoded);
    
    let out_path = format!("{}_dec.wav", path.strip_suffix(".acmp").unwrap_or(path));
    println!("Écriture du WAV décompressé vers : {}", out_path);
    write_wav(&out_path, &reconstructed_samples, sample_rate, channels).unwrap_or_else(|e| {
        eprintln!("Erreur lors de l'écriture du WAV : {}", e);
        std::process::exit(1);
    });
    
    println!("Terminé ! Le fichier est de nouveau écoutable et identique au bit près.");
}

// Minimal WAV writer
fn write_wav(path: &str, samples: &[i16], sample_rate: u32, channels: u16) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut buf = BufWriter::new(file);

    let data_size = (samples.len() * 2) as u32;
    let fmt_chunk_size: u32 = 16;
    let file_size: u32 = 4 + (8 + fmt_chunk_size) + (8 + data_size);

    // RIFF header
    buf.write_all(b"RIFF")?;
    buf.write_all(&file_size.to_le_bytes())?;
    buf.write_all(b"WAVE")?;

    // fmt chunk
    buf.write_all(b"fmt ")?;
    buf.write_all(&fmt_chunk_size.to_le_bytes())?;
    buf.write_all(&1u16.to_le_bytes())?; // format tag (PCM)
    buf.write_all(&channels.to_le_bytes())?;
    buf.write_all(&sample_rate.to_le_bytes())?;
    let byte_rate = sample_rate * channels as u32 * 2;
    buf.write_all(&byte_rate.to_le_bytes())?;
    let block_align = channels * 2;
    buf.write_all(&block_align.to_le_bytes())?;
    buf.write_all(&16u16.to_le_bytes())?; // bits per sample

    // data chunk
    buf.write_all(b"data")?;
    buf.write_all(&data_size.to_le_bytes())?;
    
    // write PCM data
    // on a little-endian machine, slice casting is faster but we iterate for safety
    for &s in samples {
        buf.write_all(&s.to_le_bytes())?;
    }
    
    buf.flush()?;
    Ok(())
}
