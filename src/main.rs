mod ingest;
mod lpc;

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

                if i < 3 || i == frame_count - 1 {
                    let ratio = if sig_e > 0.0 { res_e / sig_e } else { 0.0 };
                    println!(
                        "  Frame [{i:>5}] : coeffs=[{:.4}, {:.4}, {:.4}, ...] | erreur_pred={:.1} | énergie résidu/signal={:.4}",
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

    println!();
    if total_signal_energy > 0.0 {
        let global_ratio = total_residual_energy / total_signal_energy;
        println!(
            "Résumé : {analyzed}/{frame_count} frames analysées | ratio énergie résidu/signal global = {global_ratio:.6}"
        );
        println!(
            "         → réduction de prédiction = {:.2} dB",
            -10.0 * global_ratio.log10()
        );
    } else {
        println!("Résumé : signal nul (silence total)");
    }
}
