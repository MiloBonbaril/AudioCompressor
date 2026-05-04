//! # Entropy — L'Étau Entropique (Golomb-Rice)
//!
//! Phase 3 du compresseur audio :
//!
//! 1. **BitWriter Custom** : Buffer d'écriture bit à bit très rapide travaillant 
//!    sur des registres `u64` via des décalages logiques.
//! 2. **Analyse de Variance** : Calcul du diviseur optimal de Golomb-Rice basé
//!    sur la moyenne absolue des résidus.
//! 3. **Encodage Golomb-Rice** : Compression des résidus via ZigZag + quotient 
//!    unitaire + reste binaire.

pub struct BitWriter {
    pub buffer: Vec<u8>,
    accumulator: u64,
    bits_in_acc: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(8192), // Pré-allocation généreuse
            accumulator: 0,
            bits_in_acc: 0,
        }
    }

    /// Écrit `count` bits dans le buffer. `count` max = 64.
    #[inline(always)]
    pub fn write_bits(&mut self, value: u64, mut count: u8) {
        if count == 0 {
            return;
        }

        // Masquage de sécurité
        let mask = if count == 64 { u64::MAX } else { (1u64 << count) - 1 };
        let value = value & mask;

        // Si on dépasse la capacité du registre
        if self.bits_in_acc + count >= 64 {
            let space = 64 - self.bits_in_acc;
            let high_bits = if space == 64 { value } else { value >> (count - space) };
            
            self.accumulator |= high_bits;
            self.buffer.extend_from_slice(&self.accumulator.to_be_bytes());
            
            self.accumulator = 0;
            count -= space;
            self.bits_in_acc = 0;
        }

        // Écriture des bits restants
        if count > 0 {
            self.accumulator |= value << (64 - self.bits_in_acc - count);
            self.bits_in_acc += count;
        }
    }

    /// Écrit `q` bits à 1, suivis d'un bit à 0.
    #[inline(always)]
    pub fn write_unary(&mut self, mut q: u32) {
        // Paquets de 64 bits si q est énorme (très rare avec un bon k)
        while q >= 64 {
            self.write_bits(u64::MAX, 64);
            q -= 64;
        }
        
        if q > 0 {
            let ones = if q == 64 { u64::MAX } else { (1u64 << q) - 1 };
            self.write_bits(ones, q as u8);
        }
        
        // Le 0 terminal
        self.write_bits(0, 1);
    }

    /// Aligne sur l'octet le plus proche et vide l'accumulateur.
    pub fn flush(&mut self) {
        if self.bits_in_acc > 0 {
            let bytes = (self.bits_in_acc + 7) / 8;
            let acc_bytes = self.accumulator.to_be_bytes();
            self.buffer.extend_from_slice(&acc_bytes[..bytes as usize]);
            self.accumulator = 0;
            self.bits_in_acc = 0;
        }
    }

    /// Retourne la taille estimée en octets.
    pub fn bytes_written(&self) -> usize {
        self.buffer.len() + ((self.bits_in_acc + 7) / 8) as usize
    }

    /// Ajoute le contenu complet d'un autre BitWriter dans celui-ci.
    /// Indispensable pour l'agrégation multithread sans perte ni padding.
    pub fn append_bits(&mut self, other: &BitWriter) {
        // On récupère les octets complets du buffer
        for &byte in &other.buffer {
            self.write_bits(byte as u64, 8);
        }
        
        // On récupère les bits résiduels bloqués dans l'accumulateur
        if other.bits_in_acc > 0 {
            // L'accumulateur stocke les bits alignés à gauche (MSB).
            // Pour extraire la valeur brute, on décale vers la droite.
            let val = other.accumulator >> (64 - other.bits_in_acc);
            self.write_bits(val, other.bits_in_acc);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Golomb-Rice Encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Encodage ZigZag : map les entiers signés vers des entiers non-signés sans perte.
/// 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, etc.
#[inline(always)]
pub fn zigzag_encode(val: i32) -> u32 {
    ((val << 1) ^ (val >> 31)) as u32
}

/// Déduit instantanément le paramètre Rice `k` optimal en fonction de la moyenne.
pub fn optimal_rice_parameter(zigzag_values: &[u32]) -> u8 {
    if zigzag_values.is_empty() {
        return 0;
    }

    let mut sum: u64 = 0;
    for &z in zigzag_values {
        sum += z as u64;
    }

    let mean = sum as f64 / zigzag_values.len() as f64;
    
    // Si le signal est quasi-nul
    if mean < 1.0 {
        return 0;
    }

    // Formule analytique pour distribution de type Laplace/Géométrique : k ≈ log2(mean * ln(2))
    let k = (mean * std::f64::consts::LN_2).log2().max(0.0).round() as u8;
    k.min(31) // on évite de dépasser u32::MAX
}

/// Compresse un vecteur de résidus dans le BitWriter.
/// Retourne `(paramètre k utilisé, nombre de bits écrits)`.
pub fn encode_frame(residuals: &[f32], writer: &mut BitWriter) -> (u8, usize) {
    let start_bits = writer.buffer.len() * 8 + writer.bits_in_acc as usize;

    // 1. Casting vers i32 puis ZigZag
    let mut zigzags = Vec::with_capacity(residuals.len());
    for &r in residuals {
        zigzags.push(zigzag_encode(r.round() as i32));
    }

    // 2. Analyse de variance (moyenne absolue)
    let k = optimal_rice_parameter(&zigzags);

    // Écriture du paramètre K dans le flux pour le décodeur (sur 5 bits, max 31)
    writer.write_bits(k as u64, 5);

    // 3. Compression Golomb-Rice : bousculade de bits
    for z in zigzags {
        let q = z >> k;
        let rem = z & ((1 << k) - 1);

        writer.write_unary(q);
        writer.write_bits(rem as u64, k);
    }

    let end_bits = writer.buffer.len() * 8 + writer.bits_in_acc as usize;
    (k, end_bits - start_bits)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitwriter_basic() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b101, 3);
        bw.write_bits(0b0110, 4);
        bw.flush();
        // 1010 1100 -> 0xAC
        assert_eq!(bw.buffer[0], 0xAC);
    }

    #[test]
    fn test_bitwriter_cross_boundary() {
        let mut bw = BitWriter::new();
        // Remplir 62 bits
        bw.write_bits(u64::MAX, 62);
        // Écrire 4 bits -> 2 bits vont dans le premier u64, 2 dans le suivant
        bw.write_bits(0b1011, 4);
        bw.flush();
        
        // Les 8 premiers octets doivent être tous des 1
        for i in 0..7 {
            assert_eq!(bw.buffer[i], 0xFF);
        }
        // Le 8ème octet a reçu les 62 bits à 1, puis les bits '10' du 0b1011 (1111_1110 = 0xFE)
        assert_eq!(bw.buffer[7], 0xFE); 
        // Le 9ème octet a reçu les bits '11' du 0b1011, alignés à gauche (1100_0000 = 0xC0)
        assert_eq!(bw.buffer[8], 0xC0);
    }

    #[test]
    fn test_bitwriter_append() {
        let mut bw1 = BitWriter::new();
        bw1.write_bits(0b101, 3);
        
        let mut bw2 = BitWriter::new();
        bw2.write_bits(0b0110, 4);
        
        bw1.append_bits(&bw2);
        bw1.flush();
        
        assert_eq!(bw1.buffer[0], 0xAC); // 1010 1100
    }

    #[test]
    fn test_bitwriter_unary() {
        let mut bw = BitWriter::new();
        bw.write_unary(3); // 1110
        bw.write_unary(0); // 0
        bw.write_unary(1); // 10
        bw.flush();
        // 1110 0100 -> 0xE4
        assert_eq!(bw.buffer[0], 0xE4);
    }

    #[test]
    fn test_zigzag() {
        assert_eq!(zigzag_encode(0), 0);
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
        assert_eq!(zigzag_encode(-2), 3);
        assert_eq!(zigzag_encode(2), 4);
    }
    
    #[test]
    fn test_rice_parameter() {
        // Un signal très faible devrait avoir k=0
        let zeros = vec![0; 100];
        assert_eq!(optimal_rice_parameter(&zeros), 0);
        
        // Signal avec mean = 10 -> k=3
        let tens = vec![10; 100];
        assert_eq!(optimal_rice_parameter(&tens), 3);
        
        // Signal avec mean = 100 -> k=6
        let hundreds = vec![100; 100];
        assert_eq!(optimal_rice_parameter(&hundreds), 6);
    }

    #[test]
    fn test_encode_frame() {
        let residuals = vec![0.0, -1.0, 1.0, -2.0, 2.0];
        let mut bw = BitWriter::new();
        let (k, bits) = encode_frame(&residuals, &mut bw);
        
        // Moyenne des zigzags = (0 + 1 + 2 + 3 + 4) / 5 = 2.
        // log2(2 * ln2) = log2(1.38) ≈ 0.46, round() -> 0.
        assert_eq!(k, 0); 
        // Si k=0, bits = 5 (param k) + 1 (q=0) + 2 (q=1) + 3 (q=2) + 4 (q=3) + 5 (q=4) = 20 bits.
        assert_eq!(bits, 20);
    }
}
