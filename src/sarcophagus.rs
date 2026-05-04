//! # Sarcophage — Sérialisation Binaire
//!
//! Phase 5 du compresseur :
//!
//! 1. **Header Custom** : Un en-tête ultra-compact (22 octets) qui contient 
//!    toutes les informations de restitution.
//! 2. **Écriture Disque** : Flush massif via `BufWriter` pour anéantir 
//!    le coût des appels système (syscalls).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// En-tête de notre format binaire personnalisé.
#[derive(Debug)]
pub struct ArchiveHeader {
    pub sample_rate: u32,
    pub channels: u16,
    pub frame_size: u32,
    pub lpc_order: u8,
    pub total_frames: u32,
    pub data_size: u64,
    pub total_samples: u64,
}

/// Écrit l'archive complète sur le disque dur.
///
/// Format binaire (Little Endian) :
/// [0..4]   Magic Number "ACMP"
/// [4..5]   Version (1)
/// [5..9]   Sample Rate (u32)
/// [9..11]  Channels (u16)
/// [11..15] Frame Size (u32)
/// [15..16] LPC Order (u8)
/// [16..20] Total Frames (u32)
/// [20..28] Data Size en octets (u64)
/// [28..36] Total Samples (u64)
/// [36..]   Payload binaire compressée (BitWriter.buffer)
pub fn write_archive(
    path: impl AsRef<Path>,
    header: &ArchiveHeader,
    bitstream: &[u8],
) -> std::io::Result<()> {
    let file = File::create(path)?;
    // BufWriter massif (1 MB) pour écrire le payload d'un seul bloc (ou très peu)
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);
    
    // Magic Number & Version
    writer.write_all(b"ACMP")?;
    writer.write_all(&[1u8])?;
    
    // Métadonnées (Little Endian strict pour la portabilité)
    writer.write_all(&header.sample_rate.to_le_bytes())?;
    writer.write_all(&header.channels.to_le_bytes())?;
    writer.write_all(&header.frame_size.to_le_bytes())?;
    writer.write_all(&header.lpc_order.to_le_bytes())?;
    writer.write_all(&header.total_frames.to_le_bytes())?;
    writer.write_all(&header.data_size.to_le_bytes())?;
    writer.write_all(&header.total_samples.to_le_bytes())?;
    
    // Payload brutal
    writer.write_all(bitstream)?;
    
    writer.flush()?;
    Ok(())
}
