pub mod audio;
pub mod ml;
pub mod nlp;
pub mod storage;

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum AppError {
    AudioProcessingError(String),
    TranscriptionError(String),
    StorageError(String),
    NLPError(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AppError::AudioProcessingError(msg) => write!(f, "Audio processing error: {}", msg),
            AppError::TranscriptionError(msg) => write!(f, "Transcription error: {}", msg),
            AppError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            AppError::NLPError(msg) => write!(f, "NLP error: {}", msg),
        }
    }
}

impl Error for AppError {}
