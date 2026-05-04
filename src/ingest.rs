//! # Ingest — Zero-Copy Audio Ingestion
//!
//! Ce module implémente la Phase 1 : "L'Ingurgitation Zero-Copy".
//!
//! 1. **Mapping mémoire** : `mmap` via `memmap2` — le fichier WAV est projeté
//!    directement dans l'espace d'adressage virtuel du processus. Zéro allocation,
//!    zéro `.clone()`.
//!
//! 2. **Casting brutal** : Les octets bruts du data chunk sont réinterprétés en
//!    `&[i16]` (PCM 16-bit) ou `&[f32]` (IEEE float 32-bit) sans copie.
//!
//! 3. **Framing** : Découpage virtuel du slice en frames de taille fixe via
//!    `.chunks()` / `.chunks_exact()`. Chaque frame est un sous-slice du mmap —
//!    toujours zero-copy.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum IngestError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Not a RIFF/WAVE file (magic bytes mismatch)")]
    NotWave,

    #[error("Unsupported WAV format tag: {0} (expected 1=PCM or 3=IEEE float)")]
    UnsupportedFormat(u16),

    #[error("Unsupported bits per sample: {0}")]
    UnsupportedBitsPerSample(u16),

    #[error("WAV file truncated: expected at least {expected} bytes, got {actual}")]
    Truncated { expected: usize, actual: usize },

    #[error("'data' chunk not found in WAV file")]
    DataChunkNotFound,
}

// ─────────────────────────────────────────────────────────────────────────────
// WAV header parsing (minimal, zero-alloc)
// ─────────────────────────────────────────────────────────────────────────────

/// Format tag constants.
const WAV_FORMAT_PCM: u16 = 1;
const WAV_FORMAT_IEEE_FLOAT: u16 = 3;

/// Minimal parsed WAV metadata — everything we need, nothing we don't.
#[derive(Debug, Clone, Copy)]
pub struct WavHeader {
    pub format_tag: u16,
    pub channels: u16,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
    /// Byte offset where the audio data starts (beginning of 'data' chunk payload).
    pub data_offset: usize,
    /// Size in bytes of the audio data payload.
    pub data_size: usize,
}

/// Parse le header WAV depuis un slice d'octets mmap-é.
///
/// On navigue manuellement les chunks RIFF pour trouver 'fmt ' et 'data'.
/// Pas de dépendance externe, pas d'allocation.
fn parse_wav_header(bytes: &[u8]) -> Result<WavHeader, IngestError> {
    if bytes.len() < 12 {
        return Err(IngestError::Truncated {
            expected: 12,
            actual: bytes.len(),
        });
    }

    // RIFF header
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(IngestError::NotWave);
    }

    let mut format_tag: Option<u16> = None;
    let mut channels: Option<u16> = None;
    let mut sample_rate: Option<u32> = None;
    let mut bits_per_sample: Option<u16> = None;
    let mut data_offset: Option<usize> = None;
    let mut data_size: Option<usize> = None;

    // Walk RIFF sub-chunks starting at offset 12.
    let mut pos = 12usize;

    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;

        let chunk_data_start = pos + 8;

        match chunk_id {
            b"fmt " => {
                if chunk_data_start + 16 > bytes.len() {
                    return Err(IngestError::Truncated {
                        expected: chunk_data_start + 16,
                        actual: bytes.len(),
                    });
                }
                let d = &bytes[chunk_data_start..];
                format_tag = Some(u16::from_le_bytes([d[0], d[1]]));
                channels = Some(u16::from_le_bytes([d[2], d[3]]));
                sample_rate = Some(u32::from_le_bytes([d[4], d[5], d[6], d[7]]));
                // bytes 8..11 = byte rate, 12..13 = block align — skip
                bits_per_sample = Some(u16::from_le_bytes([d[14], d[15]]));
            }
            b"data" => {
                data_offset = Some(chunk_data_start);
                data_size = Some(chunk_size);
            }
            _ => { /* skip unknown chunks (LIST, fact, etc.) */ }
        }

        // Advance to next chunk (chunks are 2-byte aligned in RIFF).
        pos = chunk_data_start + ((chunk_size + 1) & !1);
    }

    let format_tag = format_tag.ok_or(IngestError::DataChunkNotFound)?;
    let data_offset = data_offset.ok_or(IngestError::DataChunkNotFound)?;
    let data_size = data_size.ok_or(IngestError::DataChunkNotFound)?;

    Ok(WavHeader {
        format_tag,
        channels: channels.unwrap_or(1),
        sample_rate: sample_rate.unwrap_or(44100),
        bits_per_sample: bits_per_sample.unwrap_or(16),
        data_offset,
        data_size,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Sample format — typed view over raw bytes
// ─────────────────────────────────────────────────────────────────────────────

/// Vue typée zero-copy sur les samples audio.
#[derive(Debug)]
pub enum SampleSlice<'a> {
    /// PCM 16-bit signed integers.
    I16(&'a [i16]),
    /// IEEE 754 32-bit floats.
    F32(&'a [f32]),
}

impl<'a> SampleSlice<'a> {
    /// Nombre total de samples (tous canaux confondus).
    pub fn len(&self) -> usize {
        match self {
            SampleSlice::I16(s) => s.len(),
            SampleSlice::F32(s) => s.len(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Framing iterator — zero-copy chunking
// ─────────────────────────────────────────────────────────────────────────────



// ─────────────────────────────────────────────────────────────────────────────
// MappedAudio — the zero-copy beast
// ─────────────────────────────────────────────────────────────────────────────

/// Audio file mapped en mémoire — zero-copy du fichier jusqu'aux frames.
///
/// Le `Mmap` est propriétaire du mapping ; tant que `MappedAudio` vit,
/// les slices typés qu'il expose sont valides.
pub struct MappedAudio {
    /// Le mapping mémoire — doit rester vivant tant qu'on lit les samples.
    _mmap: Mmap,
    /// Header WAV parsé.
    pub header: WavHeader,
    /// Pointeur brut + longueur vers la zone data (pour reconstruire les slices).
    data_ptr: *const u8,
    data_len: usize,
}

// SAFETY: Mmap is Send+Sync, and we only hand out shared references.
unsafe impl Send for MappedAudio {}
unsafe impl Sync for MappedAudio {}

impl MappedAudio {
    /// Ouvre et mappe un fichier WAV en mémoire.
    ///
    /// Zéro copie : le kernel fait du demand-paging, les pages sont chargées
    /// en RAM uniquement quand on y accède.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IngestError> {
        let file = File::open(path.as_ref())?;

        // SAFETY: we treat the mapping as immutable (&[u8]) for its entire lifetime.
        let mmap = unsafe { Mmap::map(&file)? };

        let header = parse_wav_header(&mmap)?;

        // Validate that the data region fits inside the mapping.
        let data_end = header.data_offset + header.data_size;
        if data_end > mmap.len() {
            return Err(IngestError::Truncated {
                expected: data_end,
                actual: mmap.len(),
            });
        }

        // Validate format.
        match header.format_tag {
            WAV_FORMAT_PCM => {
                if header.bits_per_sample != 16 {
                    return Err(IngestError::UnsupportedBitsPerSample(header.bits_per_sample));
                }
            }
            WAV_FORMAT_IEEE_FLOAT => {
                if header.bits_per_sample != 32 {
                    return Err(IngestError::UnsupportedBitsPerSample(header.bits_per_sample));
                }
            }
            other => return Err(IngestError::UnsupportedFormat(other)),
        }

        let data_ptr = mmap[header.data_offset..].as_ptr();
        let data_len = header.data_size;

        Ok(Self {
            _mmap: mmap,
            header,
            data_ptr,
            data_len,
        })
    }

    /// Casting brutal : réinterprète les octets bruts en slice typé.
    ///
    /// Zero-copy. Le slice retourné pointe directement dans le mmap.
    pub fn samples(&self) -> SampleSlice<'_> {
        match self.header.format_tag {
            WAV_FORMAT_PCM => {
                let sample_count = self.data_len / std::mem::size_of::<i16>();
                // SAFETY:
                // - `data_ptr` is valid for `data_len` bytes (validated in `open`).
                // - WAV PCM 16-bit data is always little-endian i16, and we run on LE.
                // - The mmap is immutable and outlives the returned slice.
                // - Alignment: mmap pages are page-aligned, and data_offset for
                //   standard WAV is 44 bytes (even), so alignment is satisfied.
                //   We assert this at runtime just in case.
                let ptr = self.data_ptr as *const i16;
                assert!(
                    (ptr as usize) % std::mem::align_of::<i16>() == 0,
                    "data region is not aligned for i16"
                );
                SampleSlice::I16(unsafe { std::slice::from_raw_parts(ptr, sample_count) })
            }
            WAV_FORMAT_IEEE_FLOAT => {
                let sample_count = self.data_len / std::mem::size_of::<f32>();
                let ptr = self.data_ptr as *const f32;
                assert!(
                    (ptr as usize) % std::mem::align_of::<f32>() == 0,
                    "data region is not aligned for f32"
                );
                SampleSlice::F32(unsafe { std::slice::from_raw_parts(ptr, sample_count) })
            }
            _ => unreachable!("format validated in open()"),
        }
    }

    /// Nombre total d'échantillons (tous canaux confondus).
    pub fn sample_count(&self) -> usize {
        self.samples().len()
    }

    /// Nombre de frames pour une taille de frame donnée.
    pub fn frame_count(&self, frame_size: usize) -> usize {
        (self.sample_count() + frame_size - 1) / frame_size
    }



    /// Durée totale en secondes.
    pub fn duration_secs(&self) -> f64 {
        let total_samples = self.sample_count() / self.header.channels as usize;
        total_samples as f64 / self.header.sample_rate as f64
    }
}

impl std::fmt::Debug for MappedAudio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedAudio")
            .field("header", &self.header)
            .field("data_len", &self.data_len)
            .field("sample_count", &self.sample_count())
            .field("duration_secs", &self.duration_secs())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Génère un fichier WAV PCM 16-bit minimal en mémoire.
    fn make_test_wav(samples: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
        let data_size = (samples.len() * 2) as u32;
        let fmt_chunk_size: u32 = 16;
        let file_size: u32 = 4 + (8 + fmt_chunk_size) + (8 + data_size);

        let mut buf = Vec::with_capacity(file_size as usize + 8);

        // RIFF header
        buf.write_all(b"RIFF").unwrap();
        buf.write_all(&file_size.to_le_bytes()).unwrap();
        buf.write_all(b"WAVE").unwrap();

        // fmt chunk
        buf.write_all(b"fmt ").unwrap();
        buf.write_all(&fmt_chunk_size.to_le_bytes()).unwrap();
        buf.write_all(&WAV_FORMAT_PCM.to_le_bytes()).unwrap(); // format tag
        buf.write_all(&channels.to_le_bytes()).unwrap();
        buf.write_all(&sample_rate.to_le_bytes()).unwrap();
        let byte_rate = sample_rate * channels as u32 * 2;
        buf.write_all(&byte_rate.to_le_bytes()).unwrap();
        let block_align = channels * 2;
        buf.write_all(&block_align.to_le_bytes()).unwrap();
        buf.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample

        // data chunk
        buf.write_all(b"data").unwrap();
        buf.write_all(&data_size.to_le_bytes()).unwrap();
        for &s in samples {
            buf.write_all(&s.to_le_bytes()).unwrap();
        }

        buf
    }

    fn write_temp_wav(name: &str, data: &[u8]) -> std::path::PathBuf {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join("test_wavs");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, data).unwrap();
        path
    }

    #[test]
    fn test_open_and_sample_count() {
        let samples: Vec<i16> = (0..8192).collect();
        let wav = make_test_wav(&samples, 44100, 1);
        let path = write_temp_wav("test_basic.wav", &wav);

        let audio = MappedAudio::open(&path).unwrap();
        assert_eq!(audio.header.sample_rate, 44100);
        assert_eq!(audio.header.channels, 1);
        assert_eq!(audio.header.bits_per_sample, 16);
        assert_eq!(audio.sample_count(), 8192);
    }

    #[test]
    fn test_samples_match() {
        let samples: Vec<i16> = vec![0, 1000, -1000, i16::MAX, i16::MIN];
        let wav = make_test_wav(&samples, 48000, 1);
        let path = write_temp_wav("test_values.wav", &wav);

        let audio = MappedAudio::open(&path).unwrap();
        match audio.samples() {
            SampleSlice::I16(s) => assert_eq!(s, &samples[..]),
            SampleSlice::F32(_) => panic!("expected i16"),
        }
    }


    #[test]
    fn test_stereo_duration() {
        let samples: Vec<i16> = (0..88200).map(|i| (i % 32000) as i16).collect(); // 1 second at 44100Hz stereo
        let wav = make_test_wav(&samples, 44100, 2);
        let path = write_temp_wav("test_stereo.wav", &wav);

        let audio = MappedAudio::open(&path).unwrap();
        assert_eq!(audio.header.channels, 2);
        let duration = audio.duration_secs();
        assert!((duration - 1.0).abs() < 0.001, "expected ~1s, got {duration}");
    }

    #[test]
    fn test_not_a_wav() {
        let path = write_temp_wav("test_bad.wav", b"this is not a wav file at all");
        let err = MappedAudio::open(&path).unwrap_err();
        assert!(matches!(err, IngestError::NotWave));
    }

    #[test]
    fn test_frame_count() {
        let samples: Vec<i16> = (0..10000).collect();
        let wav = make_test_wav(&samples, 44100, 1);
        let path = write_temp_wav("test_framecount.wav", &wav);

        let audio = MappedAudio::open(&path).unwrap();
        assert_eq!(audio.frame_count(4096), 3);
        assert_eq!(audio.frame_count(10000), 1);
        assert_eq!(audio.frame_count(1), 10000);
    }
}
