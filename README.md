# Idea Capture

An AI-powered note-taking app that organizes your thoughts and ideas.

## Project Structure
- `core/`: Rust core library for audio processing, ML, and NLP
- `mobile/`: Lynx mobile app
- `ml_models/`: Machine learning models

At the moment I don't have a front-end so ignore the mobile directory
1. Install Rust: https://rustup.rs/
2. Download the following models to `ml_models/pegasus/`directory: 
1º https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/spiece.model?download=true
2º https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/config.json?download=true
3º https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/rust_model.ot
3. Download the following models to `ml_models/sentence/`directory: 
1º https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/tree/main
4. Download the following models to `ml_models/whisper/`directory: 
1º Your favourite whisper model, I used the base one for testing, make
sure to change the code in core/src/audio/mod.rs to so the name of the
file that contains the model matchs the one in the path.
!!!STEPS 2 AND 3 ARE OPTIONAL, YOU CAN ALWAYS USE THE REMOTE MODEL
