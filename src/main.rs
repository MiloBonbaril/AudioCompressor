mod entropy;
mod ingest;
mod lpc;

use entropy::{encode_frame, BitWriter};

use ingest::{Frame, MappedAudio};
use lpc::{analyze_frame, DEFAULT_ORDER};

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

    let mut total_signal_energy = 0.0f64;
    let mut total_residual_energy = 0.0f64;
    let mut analyzed = 0usize;
    
    // ── Phase 3 : L'Étau Entropique ──────────────────────────────────────
    let mut bit_writer = BitWriter::new();
    let mut total_uncompressed_bits = 0usize;
    let mut total_compressed_bits = 0usize;

    for (i, frame) in audio.frames(frame_size).enumerate() {
        let samples = match &frame {
            Frame::I16(s) => *s,
            Frame::F32(_) => {
                eprintln!("  Frame [{i}] : format f32 non supporté par le moteur LPC (i16 requis)");
                continue;
            }
        };

        if samples.len() <= DEFAULT_ORDER {
            continue; // Frame trop courte pour l'analyse LPC.
        }

        match analyze_frame(samples, DEFAULT_ORDER) {
            Some(analysis) => {
                let sig_e: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
                let res_e: f64 = analysis.residual.iter().map(|&r| (r as f64).powi(2)).sum();
                total_signal_energy += sig_e;
                total_residual_energy += res_e;
                analyzed += 1;
                
                // Encodage Golomb-Rice (Phase 3)
                let uncompressed_bits = samples.len() * 16;
                let (k, compressed_bits) = encode_frame(&analysis.residual, &mut bit_writer);
                
                total_uncompressed_bits += uncompressed_bits;
                total_compressed_bits += compressed_bits;

                if i < 3 || i == frame_count - 1 {
                    let ratio = if sig_e > 0.0 { res_e / sig_e } else { 0.0 };
                    let compression_ratio = compressed_bits as f64 / uncompressed_bits as f64 * 100.0;
                    println!(
                        "  Frame [{i:>5}] : coeffs=[{:.4}, {:.4}, {:.4}, ...] | erreur_pred={:.1} | énergie résidu/signal={:.4} | k={k} ({compression_ratio:.1}%)",
                        analysis.coefficients.coeffs[0],
                        analysis.coefficients.coeffs[1],
                        analysis.coefficients.coeffs[2],
                        analysis.coefficients.prediction_error,
                        ratio,
                    );
                } else if i == 3 {
                    println!("  ...");
                }
            }
            None => {
                if i < 3 || i == frame_count - 1 {
                    println!("  Frame [{i:>5}] : silence/instable — skip");
                }
            }
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
    } else {
        println!("Signal nul (silence total)");
    }
}
