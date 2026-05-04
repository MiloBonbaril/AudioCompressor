use rayon::prelude::*;

use audio_compressor::entropy::{encode_frame, BitWriter};
use audio_compressor::ingest::{MappedAudio, SampleSlice};
use audio_compressor::lpc::{self, DEFAULT_ORDER};
use audio_compressor::sarcophagus::{self, ArchiveHeader};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: audio_compressor <fichier.wav>");
        std::process::exit(1);
    });

    let audio = MappedAudio::open(&path).unwrap_or_else(|e| {
        eprintln!("Erreur lors de l'ouverture de {path}: {e}");
        std::process::exit(1);
    });

    println!("=== Phase 1 : Ingurgitation Zero-Copy ===");
    println!("{audio:#?}");
    println!();

    let frame_size = lpc::DEFAULT_FRAME_SIZE;
    let frame_count = audio.frame_count(frame_size);
    println!(
        "Framing : {} frames de {} samples",
        frame_count, frame_size
    );
    println!();

    // ── Phase 2 : Moteur LPC ─────────────────────────────────────────────
    println!("=== Phase 2 : Moteur LPC (ordre {DEFAULT_ORDER}) ===");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            println!("  [SIMD] AVX2 détecté — autocorrélation vectorisée active");
        } else {
            println!("  [SCALAR] AVX2 absent — fallback scalaire");
        }
    }
    println!();

    // ── Phase 4 : L'Assaut des Cœurs (Multithreading) ────────────────────
    let mut bit_writer = BitWriter::new();
    
    let i16_samples = match audio.samples() {
        SampleSlice::I16(s) => s,
        SampleSlice::F32(_) => {
            eprintln!("Format f32 non supporté par le moteur LPC (i16 requis)");
            std::process::exit(1);
        }
    };
    
    struct FrameResult {
        signal_energy: f64,
        residual_energy: f64,
        uncompressed_bits: usize,
        compressed_bits: usize,
        writer: BitWriter,
        log_line: Option<String>,
    }

    // Map-Reduce Lock-Free via rayon
    let results: Vec<FrameResult> = i16_samples
        .par_chunks(frame_size)
        .enumerate()
        .map(|(i, samples)| {
            let mut writer = BitWriter::new();
            
            let analysis = lpc::analyze_frame(samples, DEFAULT_ORDER);
            
            // Écriture des coefficients LPC de la frame avant le résidu
            for &c in &analysis.coefficients.coeffs {
                writer.write_f64(c);
            }
            
            let sig_e: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
            let res_e: f64 = analysis.residual.iter().map(|&r| (r as f64).powi(2)).sum();
            
            let uncompressed_bits = samples.len() * 16;
            let (k, compressed_bits) = encode_frame(&analysis.residual, &mut writer);
            
            let log_line = if i < 3 || i == frame_count - 1 {
                let ratio = if sig_e > 0.0 { res_e / sig_e } else { 0.0 };
                let compression_ratio = compressed_bits as f64 / uncompressed_bits as f64 * 100.0;
                Some(format!(
                    "  Frame [{i:>5}] : coeffs=[{:.4}, {:.4}, {:.4}, ...] | erreur_pred={:.1} | énergie résidu/signal={:.4} | k={k} ({compression_ratio:.1}%)",
                    analysis.coefficients.coeffs.get(0).unwrap_or(&0.0),
                    analysis.coefficients.coeffs.get(1).unwrap_or(&0.0),
                    analysis.coefficients.coeffs.get(2).unwrap_or(&0.0),
                    analysis.coefficients.prediction_error,
                    ratio
                ))
            } else if i == 3 {
                Some("  ...".to_string())
            } else {
                None
            };

            FrameResult {
                signal_energy: sig_e,
                residual_energy: res_e,
                uncompressed_bits,
                compressed_bits,
                writer,
                log_line,
            }
        })
        .collect();

    // Agrégation Séquentielle Déterministe
    let mut total_signal_energy = 0.0f64;
    let mut total_residual_energy = 0.0f64;
    let mut total_uncompressed_bits = 0usize;
    let mut total_compressed_bits = 0usize;
    let mut analyzed = 0usize;

    for res in results {
        if let Some(line) = res.log_line {
            println!("{}", line);
        }
        
        if res.uncompressed_bits > 0 {
            analyzed += 1;
            total_signal_energy += res.signal_energy;
            total_residual_energy += res.residual_energy;
            total_uncompressed_bits += res.uncompressed_bits;
            total_compressed_bits += res.compressed_bits;
            bit_writer.append_bits(&res.writer);
        }
    }
    
    // Aligner le dernier octet du BitWriter
    bit_writer.flush();

    println!();
    println!("=== Résumé ===");
    if total_signal_energy > 0.0 {
        let global_ratio = total_residual_energy / total_signal_energy;
        let global_compression = total_compressed_bits as f64 / total_uncompressed_bits as f64 * 100.0;
        
        println!(
            "Analysé        : {analyzed}/{frame_count} frames"
        );
        println!(
            "Moteur LPC     : Réduction de l'énergie de prédiction = {:.2} dB",
            -10.0 * global_ratio.log10()
        );
        println!(
            "Entropie (Rice): Compression à {:.1}% de la taille brute (Ratio final: {:.2}x)",
            global_compression,
            total_uncompressed_bits as f64 / total_compressed_bits as f64
        );
        println!(
            "BitWriter      : Buffer final de {} octets générés",
            bit_writer.bytes_written()
        );
        
        // ── Phase 5 : Le Sarcophage Binaire ───────────────────────────────
        let compressed_path = format!("{}.acmp", path);
        let header = ArchiveHeader {
            sample_rate: audio.header.sample_rate,
            channels: audio.header.channels,
            frame_size: frame_size as u32,
            lpc_order: DEFAULT_ORDER as u8,
            total_frames: frame_count as u32,
            data_size: bit_writer.buffer.len() as u64,
            total_samples: audio.sample_count() as u64,
        };
        
        match sarcophagus::write_archive(&compressed_path, &header, &bit_writer.buffer) {
            Ok(_) => println!("Sarcophage     : Fichier écrit avec succès dans {}", compressed_path),
            Err(e) => eprintln!("Sarcophage     : Erreur d'écriture -> {}", e),
        }
    } else {
        println!("Signal nul (silence total)");
    }
}
