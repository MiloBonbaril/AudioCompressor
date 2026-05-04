mod ingest;

use ingest::MappedAudio;

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

    let frame_size = 4096;
    let frame_count = audio.frame_count(frame_size);
    println!(
        "Framing : {} frames de {} samples (dernière frame potentiellement plus courte)",
        frame_count, frame_size
    );

    // Itère les frames sans jamais copier un seul octet.
    for (i, frame) in audio.frames(frame_size).enumerate() {
        if i < 3 || i == frame_count - 1 {
            println!("  Frame [{i:>5}] : {} samples", frame.len());
        } else if i == 3 {
            println!("  ...");
        }
    }
}
