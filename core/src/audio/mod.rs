//! Audio processing module for Idea Capture
//! Handles recording, preprocessing, preparing audio for transcription, and playback
use crate::AppError;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound;
use std::env;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use whisper_rs::{FullParams, WhisperContext};

pub struct AudioProcessor {
    sample_rate: u32,
    channels: u16,
    is_recording: Arc<AtomicBool>,
    whisper_context: Option<Arc<WhisperContext>>,
}

impl AudioProcessor {
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self, AppError> {
        let model_path = env::var("MODEL_PATH");
        let whisper_path = model_path.unwrap() + "whisper/whisper-medium.bin";
        let ctx = WhisperContext::new(&whisper_path)
            .map_err(|e| AppError::TranscriptionError(e.to_string()))?;

        Ok(Self {
            sample_rate,
            channels,
            is_recording: Arc::new(AtomicBool::new(false)),
            whisper_context: Some(Arc::new(ctx)),
        })
    }

    /// Record audio from microphone for specified duration
    pub fn start_recording(&self) -> Result<RecordingHandle, AppError> {
        if self.is_recording.load(Ordering::Relaxed) {
            return Err(AppError::AudioProcessingError(
                "Already recording".to_string(),
            ));
        }
        let host = cpal::default_host();

        let device = host.default_input_device().ok_or_else(|| {
            AppError::AudioProcessingError("No input device available".to_string())
        })?;

        let mut supported_configs = device
            .supported_input_configs()
            .map_err(|e| AppError::AudioProcessingError(e.to_string()))?;

        let supported_config = supported_configs
            .find(|config| {
                config.channels() == self.channels && config.max_sample_rate().0 >= self.sample_rate
            })
            .ok_or_else(|| AppError::AudioProcessingError("No supported config found".to_string()))?
            .with_sample_rate(cpal::SampleRate(self.sample_rate));

        let actual_sample_rate = supported_config.sample_rate().0;
        println!("Recording at sample rate: {}", actual_sample_rate);

        let samples = Arc::new(Mutex::new(Vec::new()));
        let samples_clone = samples.clone();

        self.is_recording.store(true, Ordering::Relaxed);

        let stream = device
            .build_input_stream(
                &supported_config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut buffer = samples_clone.lock().unwrap();
                    buffer.extend_from_slice(data);
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )
            .map_err(|e| AppError::AudioProcessingError(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AppError::AudioProcessingError(e.to_string()))?;

        Ok(RecordingHandle {
            stream,
            samples,
            is_recording: self.is_recording.clone(),
        })
    }

    pub fn transcribe(&self, audio_data: &[f32]) -> Result<String, AppError> {
        let mut state = self
            .whisper_context
            .as_ref()
            .unwrap()
            .create_state()
            .map_err(|e| AppError::TranscriptionError(e.to_string()))?;

        let mut params = FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });

        params.set_language(Some("en"));

        state
            .full(params, &audio_data)
            .map_err(|e| AppError::TranscriptionError(e.to_string()))?;

        let transcript = (0..state
            .full_n_segments()
            .map_err(|e| AppError::TranscriptionError(e.to_string()))?)
            .map(|i| state.full_get_segment_text(i))
            .collect::<Result<String, _>>()
            .map_err(|e| AppError::TranscriptionError(e.to_string()))?;
        Ok(transcript)
    }

    /// Play back recorded audio samples
    pub fn play_audio(&self, samples: Vec<f32>) -> Result<(), AppError> {
        let host = cpal::default_host();

        let device = host.default_output_device().ok_or_else(|| {
            AppError::AudioProcessingError("No output device available".to_string())
        })?;

        let mut supported_configs = device
            .supported_output_configs()
            .map_err(|e| AppError::AudioProcessingError(e.to_string()))?;

        let supported_config = supported_configs
            .find(|config| {
                config.channels() == self.channels && config.max_sample_rate().0 >= self.sample_rate
            })
            .ok_or_else(|| {
                AppError::AudioProcessingError("No supported output config found".to_string())
            })?
            .with_sample_rate(cpal::SampleRate(self.sample_rate));

        println!(
            "Playing at sample rate: {}",
            supported_config.sample_rate().0
        );

        let samples = Arc::new(Mutex::new(samples.into_iter().collect::<Vec<f32>>()));
        let samples_clone = samples.clone();

        let stream = device
            .build_output_stream(
                &supported_config.into(),
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut input = samples_clone.lock().unwrap();
                    let len = data.len().min(input.len());
                    data[..len].copy_from_slice(&input[..len]);
                    input.drain(..len); // Remove played samples
                },
                |err| eprintln!("Audio playback error: {}", err),
                Some(Duration::from_secs(5)),
            )
            .map_err(|e| AppError::AudioProcessingError(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AppError::AudioProcessingError(e.to_string()))?;

        let duration =
            Duration::from_secs_f32(samples.lock().unwrap().len() as f32 / self.sample_rate as f32);
        std::thread::sleep(duration);
        drop(stream);
        Ok(())
    }

    /// Load audio from a WAV file
    pub fn load_audio(&self, path: &str) -> Result<Vec<f32>, AppError> {
        let reader = hound::WavReader::open(path).map_err(|e| {
            AppError::AudioProcessingError(format!("Failed to open WAV file: {}", e))
        })?;

        let samples: Vec<f32> = reader
            .into_samples::<i16>()
            .map(|s| s.unwrap_or(0) as f32 / 32768.0)
            .collect();

        Ok(samples)
    }

    /// Apply basic noise reduction to audio samples
    pub fn reduce_noise(&self, samples: &[f32]) -> Vec<f32> {
        let threshold = 0.02;
        samples
            .iter()
            .map(|&s| if s.abs() < threshold { 0.0 } else { s })
            .collect()
    }
}

pub struct RecordingHandle {
    stream: cpal::Stream,
    samples: Arc<Mutex<Vec<f32>>>,
    is_recording: Arc<AtomicBool>,
}

impl RecordingHandle {
    pub fn stop_recording(self) -> Result<Vec<f32>, AppError> {
        self.is_recording.store(false, Ordering::Relaxed);

        self.stream
            .pause()
            .map_err(|e| AppError::AudioProcessingError(e.to_string()))?;

        let recorded_samples = {
            let locked_samples = self.samples.lock().unwrap();
            locked_samples.clone()
        };

        Ok(recorded_samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_reduction() {
        let processor = AudioProcessor::new(44100, 1).unwrap();
        let samples = vec![0.01, 0.03, -0.01, 0.5, -0.6, 0.015];
        let processed = processor.reduce_noise(&samples);
        assert_eq!(processed, vec![0.0, 0.03, 0.0, 0.5, -0.6, 0.0]);
    }
    #[test]
    fn test_record_and_transcribe() {
        let processor = AudioProcessor::new(16000, 1).unwrap();

        println!("Please say 'hello world' in the next 5 seconds...");

        // Start recording
        let recording_handle = processor
            .start_recording()
            .expect("Failed to start recording");

        use std::io::{stdin, Read};
        let mut stdin = stdin();
        let _ = stdin.read(&mut [0u8]).expect("Failed to read from stdin");

        // Stop recording and get samples
        let samples = recording_handle
            .stop_recording()
            .expect("Failed to stop recording");

        // Process, transcribe and play back
        let cleaned_samples = processor.reduce_noise(&samples);
        let transcription = processor
            .transcribe(&cleaned_samples)
            .expect("Transcription failed");
        println!("Transcription: '{}'", transcription);

        // Play back to verify what was recorded
        println!("Playing back recorded audio...");
        processor.play_audio(samples).expect("Playback failed");
    }
}
