//! # LPC Engine — Vectorisation Ciblée
//!
//! Phase 2 du compresseur audio :
//!
//! 1. **Autocorrélation SIMD** : Intrinsèques AVX2, 16 échantillons i16 par
//!    `_mm256_madd_epi16`, accumulation i64 pour éviter tout overflow.
//!    Fallback scalaire automatique si AVX2 absent.
//!
//! 2. **Levinson-Durbin** : Résolution du système de Toeplitz, bridé à un
//!    ordre LPC configurable (défaut 10, suffisant pour du 24 kHz).
//!
//! 3. **Génération du résidu** : Filtre FIR inverse appliqué sur la frame
//!    pour produire le vecteur d'erreurs de prédiction.

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Ordre LPC par défaut — règle du pouce : sample_rate/1000 + 2 ≈ 10 pour 24 kHz.
pub const DEFAULT_ORDER: usize = 10;

/// Taille de frame par défaut.
pub const DEFAULT_FRAME_SIZE: usize = 4096;

/// Ordre LPC maximum supporté.
pub const MAX_ORDER: usize = 32;

// ─────────────────────────────────────────────────────────────────────────────
// Autocorrelation — scalar fallback
// ─────────────────────────────────────────────────────────────────────────────

/// Autocorrélation scalaire : R[k] = Σ x[n]·x[n+k], pour k = 0..=max_lag.
///
/// Sert de fallback quand AVX2 n'est pas disponible, et de référence pour les tests.
fn autocorrelation_scalar(samples: &[i16], max_lag: usize) -> Vec<f64> {
    let n = samples.len();
    let mut result = vec![0.0f64; max_lag + 1];

    for lag in 0..=max_lag {
        let mut acc: i64 = 0;
        for i in 0..n - lag {
            acc += samples[i] as i64 * samples[i + lag] as i64;
        }
        result[lag] = acc as f64;
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Autocorrelation — AVX2 SIMD
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Somme horizontale d'un vecteur 4×i64 → i64 scalaire.
///
/// SAFETY: Requires AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i64x4(v: __m256i) -> i64 {
    // SAFETY: __m256i is just 256 bits, same layout as [i64; 4].
    unsafe {
        let arr: [i64; 4] = std::mem::transmute(v);
        arr[0] + arr[1] + arr[2] + arr[3]
    }
}

/// Autocorrélation AVX2 : 16 échantillons i16 par itération.
///
/// Pipeline :
///   1. `_mm256_loadu_si256`  — charge 16×i16 (non-aligné, gratuit sur µarch modernes)
///   2. `_mm256_madd_epi16`   — multiply + horizontal add par paires → 8×i32
///   3. Widening vers 2×(4×i64) via `_mm256_cvtepi32_epi64`
///   4. Accumulation dans deux registres i64 → somme horizontale en fin de boucle
///
/// SAFETY: caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn autocorrelation_avx2(samples: &[i16], max_lag: usize) -> Vec<f64> {
    let n = samples.len();
    let mut result = vec![0.0f64; max_lag + 1];
    let base_ptr = samples.as_ptr();

    for lag in 0..=max_lag {
        let len = n - lag;
        // SAFETY: lag < n (asserted by caller), so base_ptr.add(lag) is within the slice.
        let ptr_a = base_ptr;
        let ptr_b = unsafe { base_ptr.add(lag) };

        // Accumulateurs i64 — pas de risque d'overflow même sur des frames énormes.
        let mut acc_lo = _mm256_setzero_si256(); // 4×i64
        let mut acc_hi = _mm256_setzero_si256(); // 4×i64

        let simd_end = len & !15; // len arrondi au multiple de 16 inférieur
        let mut i = 0usize;

        while i < simd_end {
            // SAFETY: i + 16 <= simd_end <= len, ptrs are within slice bounds.
            unsafe {
                let a = _mm256_loadu_si256(ptr_a.add(i) as *const __m256i);
                let b = _mm256_loadu_si256(ptr_b.add(i) as *const __m256i);

                // 16×i16 * 16×i16 → pairwise mul + add adjacent → 8×i32
                let prod = _mm256_madd_epi16(a, b);

                // Widen 8×i32 → 2×(4×i64)
                let lo_128 = _mm256_castsi256_si128(prod);
                let hi_128 = _mm256_extracti128_si256::<1>(prod);

                acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(lo_128));
                acc_hi = _mm256_add_epi64(acc_hi, _mm256_cvtepi32_epi64(hi_128));
            }

            i += 16;
        }

        // Réduction horizontale des 8 lanes i64 → 1 scalaire.
        let total_vec = _mm256_add_epi64(acc_lo, acc_hi);
        let mut total: i64 = unsafe { hsum_i64x4(total_vec) };

        // Queue scalaire (< 16 échantillons restants).
        for j in simd_end..len {
            // SAFETY: j < len = n - lag, so both ptr offsets are within the slice.
            unsafe {
                total += *ptr_a.add(j) as i64 * *ptr_b.add(j) as i64;
            }
        }

        result[lag] = total as f64;
    }

    result
}

/// Autocorrélation avec dispatch dynamique : AVX2 si disponible, sinon scalaire.
pub fn autocorrelation(samples: &[i16], max_lag: usize) -> Vec<f64> {
    assert!(
        max_lag < samples.len(),
        "max_lag ({max_lag}) must be < samples.len() ({})",
        samples.len()
    );

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { autocorrelation_avx2(samples, max_lag) };
        }
    }

    autocorrelation_scalar(samples, max_lag)
}

// ─────────────────────────────────────────────────────────────────────────────
// Levinson-Durbin solver
// ─────────────────────────────────────────────────────────────────────────────

/// Résultat de la résolution Levinson-Durbin.
#[derive(Debug, Clone)]
pub struct LpcCoefficients {
    /// Coefficients LPC a[1..=order] (a[0] est implicitement 1.0).
    pub coeffs: Vec<f64>,
    /// Ordre effectif du filtre.
    pub order: usize,
    /// Erreur de prédiction finale (variance du résidu).
    pub prediction_error: f64,
}

/// Résout le système de Toeplitz via l'algorithme de Levinson-Durbin.
///
/// Entrée : `autocorr[0..=order]` — les valeurs R[0], R[1], ..., R[order].
/// Sortie : coefficients LPC et erreur de prédiction.
///
/// Retourne `None` si :
/// - R[0] ≤ 0 (signal nul ou silence)
/// - Le processus diverge (|k_m| ≥ 1 → instabilité)
pub fn levinson_durbin(autocorr: &[f64], order: usize) -> Option<LpcCoefficients> {
    assert!(
        autocorr.len() > order,
        "autocorr must have at least order+1 elements"
    );

    if autocorr[0] <= 0.0 {
        return None; // Signal nul — pas de prédiction possible.
    }

    let mut error = autocorr[0];
    let mut a = vec![0.0f64; order + 1]; // a[0] = 1.0 implicite, on utilise a[1..=order]
    let mut a_prev = vec![0.0f64; order + 1];

    for m in 1..=order {
        // Calcul du coefficient de réflexion k_m (PARCOR).
        let mut lambda = autocorr[m];
        for k in 1..m {
            lambda -= a_prev[k] * autocorr[m - k];
        }
        lambda /= error;

        // Vérification de stabilité : |k_m| doit être < 1.
        if lambda.abs() >= 1.0 {
            return None;
        }

        // Mise à jour des coefficients.
        a[m] = lambda;
        for k in 1..m {
            a[k] = a_prev[k] - lambda * a_prev[m - k];
        }

        // Mise à jour de l'erreur de prédiction.
        error *= 1.0 - lambda * lambda;

        // Swap pour la prochaine itération.
        std::mem::swap(&mut a, &mut a_prev);
    }

    // Après la boucle, les résultats finaux sont dans a_prev (dernier swap).
    Some(LpcCoefficients {
        coeffs: a_prev[1..=order].to_vec(),
        order,
        prediction_error: error,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Residual generation
// ─────────────────────────────────────────────────────────────────────────────

/// Génère le résidu (erreur de prédiction) en appliquant le filtre LPC inverse.
///
/// ```text
/// e[n] = x[n] - Σ_{k=1}^{order} a[k] · x[n-k]
/// ```
///
/// Pour n < order, les échantillons hors-limites (x[négatif]) sont traités comme 0.
/// Le résidu a une dynamique plus faible que le signal original — c'est tout
/// l'intérêt de la prédiction linéaire pour la compression.
pub fn compute_residual(samples: &[i16], lpc: &LpcCoefficients) -> Vec<f32> {
    let n = samples.len();
    let order = lpc.order;
    let coeffs = &lpc.coeffs;
    let mut residual = Vec::with_capacity(n);

    for i in 0..n {
        let sample = samples[i] as f64;
        let mut prediction = 0.0f64;

        // Σ a[k] · x[n-k] pour k=1..order, en protégeant les bords.
        let max_k = order.min(i);
        for k in 0..max_k {
            prediction += coeffs[k] * samples[i - 1 - k] as f64;
        }

        residual.push((sample - prediction) as f32);
    }

    residual
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience — analyse complète d'une frame
// ─────────────────────────────────────────────────────────────────────────────

/// Résultat complet de l'analyse LPC d'une frame.
#[derive(Debug)]
pub struct LpcAnalysis {
    /// Coefficients LPC + erreur de prédiction.
    pub coefficients: LpcCoefficients,
    /// Résidu (erreurs de prédiction), même taille que la frame d'entrée.
    pub residual: Vec<f32>,
}

/// Analyse LPC complète d'une frame i16.
///
/// 1. Autocorrélation (AVX2 si dispo)
/// 2. Levinson-Durbin
/// 3. Génération du résidu
///
/// Retourne `None` si le signal est nul (silence) ou instable.
pub fn analyze_frame(samples: &[i16], order: usize) -> Option<LpcAnalysis> {
    assert!(order <= MAX_ORDER, "order {order} exceeds MAX_ORDER {MAX_ORDER}");
    assert!(
        samples.len() > order,
        "frame too short ({}) for order {order}",
        samples.len()
    );

    let autocorr = autocorrelation(samples, order);
    let coefficients = levinson_durbin(&autocorr, order)?;
    let residual = compute_residual(samples, &coefficients);

    Some(LpcAnalysis {
        coefficients,
        residual,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Vérifie que l'autocorrélation scalaire est correcte sur un cas trivial.
    #[test]
    fn test_autocorr_scalar_dc() {
        // Signal constant = 100 → R[0] = N * 100², R[k>0] = (N-k) * 100²
        let samples: Vec<i16> = vec![100; 64];
        let r = autocorrelation_scalar(&samples, 3);
        assert_eq!(r[0], 64.0 * 10000.0);
        assert_eq!(r[1], 63.0 * 10000.0);
        assert_eq!(r[2], 62.0 * 10000.0);
        assert_eq!(r[3], 61.0 * 10000.0);
    }

    /// Vérifie que l'autocorrélation scalaire donne R[0] = énergie.
    #[test]
    fn test_autocorr_scalar_energy() {
        let samples: Vec<i16> = (1..=100).map(|x| x as i16).collect();
        let r = autocorrelation_scalar(&samples, 0);
        let expected: i64 = (1..=100i64).map(|x| x * x).sum();
        assert_eq!(r[0], expected as f64);
    }

    /// AVX2 et scalaire doivent donner des résultats identiques.
    #[test]
    fn test_avx2_matches_scalar() {
        // Signal pseudo-aléatoire mais déterministe.
        let samples: Vec<i16> = (0..4096)
            .map(|i| ((i * 7 + 13) % 65536 - 32768) as i16)
            .collect();

        let max_lag = 10;
        let scalar = autocorrelation_scalar(&samples, max_lag);
        let dispatched = autocorrelation(&samples, max_lag);

        for lag in 0..=max_lag {
            assert!(
                (scalar[lag] - dispatched[lag]).abs() < 1e-6,
                "mismatch at lag {lag}: scalar={}, dispatched={}",
                scalar[lag],
                dispatched[lag]
            );
        }
    }

    /// Levinson-Durbin sur un AR(1) connu : x[n] = 0.9·x[n-1] + bruit.
    /// Les coefficients doivent retrouver a[1] ≈ 0.9.
    #[test]
    fn test_levinson_durbin_ar1() {
        // Autocorrélation théorique d'un AR(1) avec coeff 0.9 et variance 1 :
        //   R[k] = σ² / (1 - a²) · a^|k|
        // Avec a=0.9, σ²=1 : R[k] = 1/(1-0.81) · 0.9^k ≈ 5.263 · 0.9^k
        let a = 0.9f64;
        let var = 1.0 / (1.0 - a * a);
        let autocorr: Vec<f64> = (0..=10).map(|k| var * a.powi(k as i32)).collect();

        let lpc = levinson_durbin(&autocorr, 1).unwrap();
        assert_eq!(lpc.order, 1);
        assert!(
            (lpc.coeffs[0] - 0.9).abs() < 1e-10,
            "expected a[1]≈0.9, got {}",
            lpc.coeffs[0]
        );
    }

    /// Silence → levinson_durbin retourne None.
    #[test]
    fn test_levinson_durbin_silence() {
        let autocorr = vec![0.0; 11];
        assert!(levinson_durbin(&autocorr, 10).is_none());
    }

    /// Le résidu d'un signal constant doit être très faible après le premier échantillon.
    ///
    /// Note : avec N=256 le coefficient LPC est R[1]/R[0] = 255/256 ≈ 0.996,
    /// pas exactement 1.0. Le résidu est donc ~1000 * (1 - 255/256) ≈ 3.9.
    #[test]
    fn test_residual_constant_signal() {
        let samples: Vec<i16> = vec![1000; 256];
        let analysis = analyze_frame(&samples, 1).unwrap();

        // Premier échantillon : pas de prédiction → résidu = signal.
        assert!((analysis.residual[0] - 1000.0).abs() < 1e-3);

        // Le coefficient LPC est ~0.996, donc le résidu est ~3.9, pas 0.
        // Mais il doit être très petit par rapport au signal (1000).
        for i in 1..256 {
            assert!(
                analysis.residual[i].abs() < 5.0,
                "residual[{i}] = {} (expected < 5.0)",
                analysis.residual[i]
            );
        }
    }

    /// Analyse complète sur une frame de 4096 samples : vérifier les dimensions.
    #[test]
    fn test_analyze_frame_dimensions() {
        let samples: Vec<i16> = (0..4096)
            .map(|i| (((i as f64) * 0.1).sin() * 10000.0) as i16)
            .collect();

        let analysis = analyze_frame(&samples, DEFAULT_ORDER).unwrap();
        assert_eq!(analysis.coefficients.order, DEFAULT_ORDER);
        assert_eq!(analysis.coefficients.coeffs.len(), DEFAULT_ORDER);
        assert_eq!(analysis.residual.len(), 4096);
        assert!(analysis.coefficients.prediction_error > 0.0);
    }

    /// Vérifier que le résidu a une énergie plus faible que le signal original.
    #[test]
    fn test_residual_energy_reduction() {
        // Signal sinusoïdal = très prédictible → résidu doit avoir énergie << signal.
        let samples: Vec<i16> = (0..4096)
            .map(|i| (((i as f64) * 2.0 * std::f64::consts::PI / 100.0).sin() * 15000.0) as i16)
            .collect();

        let analysis = analyze_frame(&samples, DEFAULT_ORDER).unwrap();

        let signal_energy: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
        let residual_energy: f64 = analysis.residual.iter().map(|&r| (r as f64).powi(2)).sum();

        let ratio = residual_energy / signal_energy;
        assert!(
            ratio < 0.1,
            "residual energy ratio {ratio:.4} should be < 0.1 for a sinusoidal signal"
        );
    }
}
