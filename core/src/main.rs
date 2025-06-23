use idea_capture_core::{audio, nlp};

#[cfg(target_os = "linux")]
fn os_specific_function() {
    println!("Running on Linux!");
    // Linux-specific code here
}

#[cfg(target_os = "windows")]
fn os_specific_function() {
    println!("Running on Windows!");
    // Windows-specific code here
}

#[cfg(target_os = "macos")]
fn os_specific_function() {
    println!("Running on macOS!");
    // macOS-specific code here
}
fn main() {
    let processor = audio::AudioProcessor::new(16000, 1);

    // Start recording
    let recording_handle = processor
        .as_ref()
        .unwrap()
        .start_recording()
        .expect("Failed to start recording");

    println!("Please press a key to stop recording...");
    use std::io::{stdin, Read};
    let mut stdin = stdin();
    let _ = stdin.read(&mut [0u8]).expect("Failed to read from stdin");

    // Stop recording and get samples
    let samples = recording_handle
        .stop_recording()
        .expect("Failed to stop recording");

    // Process, transcribe and play back
    let cleaned_samples = processor.as_ref().unwrap().reduce_noise(&samples);
    let transcription = processor
        .unwrap()
        .transcribe(&cleaned_samples)
        .expect("Transcription failed");
    println!("Transcription: '{}'", transcription);

    let notes = nlp::process_audio_transcript(&transcription);
    if let Ok(notes) = notes {
        for note in notes {
            println!("{:?}", note);
        }
    }
}
