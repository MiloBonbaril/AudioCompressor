use criterion::{black_box, criterion_group, criterion_main, Criterion};
use audio_compressor::lpc::{analyze_frame, autocorrelation, DEFAULT_ORDER};

fn bench_autocorr(c: &mut Criterion) {
    // Frame synthétique
    let samples: Vec<i16> = (0..4096).map(|i| (i % 1000) as i16).collect();
    
    c.bench_function("autocorrelation_simd", |b| {
        b.iter(|| {
            // Le black_box empêche le compilateur d'optimiser l'appel
            autocorrelation(black_box(&samples), black_box(DEFAULT_ORDER))
        })
    });
}

fn bench_analyze_frame(c: &mut Criterion) {
    let samples: Vec<i16> = (0..4096).map(|i| (i % 1000) as i16).collect();
    
    c.bench_function("analyze_frame_complet", |b| {
        b.iter(|| {
            analyze_frame(black_box(&samples), black_box(DEFAULT_ORDER))
        })
    });
}

criterion_group!(benches, bench_autocorr, bench_analyze_frame);
criterion_main!(benches);
