# Idea Capture

An AI-powered note-taking app that organizes your thoughts and ideas.

## Project Structure
- `core/`: Rust core library for audio processing, ML, and NLP
- `mobile/`: React Native mobile app
- `ml_models/`: Machine learning models

## Development Setup
1. Install Rust: https://rustup.rs/
2. Install Node.js and npm
3. Set up React Native environment: https://reactnative.dev/docs/environment-setup
4. Download Whisper models to `ml_models/` directory

## Building the Core Library
```bash
cd core
cargo build --release
```

## Running the Mobile App
```bash
cd mobile
npm install
npm run dev
```
# idea-capture
