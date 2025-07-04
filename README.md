# Idea Capture

An AI-powered note-taking app that organizes your thoughts and ideas.

## Project Structure
- `core/`: Rust core library for audio processing, ML, and NLP
- `mobile/`: Lynx mobile app
- `ml_models/`: Machine learning models

## Development Setup
1. Install Rust: https://rustup.rs/
2. Install Node.js and npm
3. Set up Lynx environment: https://lynxjs.org/guide/start/quick-start
4. Download the following models to `ml_models/`directory: 
1º https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/spiece.model?download=true
2º https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/config.json?download=true
3º https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/rust_model.ot
4º NO ENCUENTRO EL PUTO ENLACE PARA DESCARGAR WHISPER-BASE.BIN
5º https://cas-bridge.xethub.hf.co/xet-bridge-us/621ffdc136468d709f180292/c703f06bebc92db4ce35dfa0bbea7c3f6e8b4b4acce635b982cc8727b8e573ef?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20250624%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250624T144202Z&X-Amz-Expires=3600&X-Amz-Signature=4304bfb0fe6d8200143eeaa18b7597b6dc3d881b62dde944e29c8fdc635b1e15&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=public&response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27rust_model.ot%3B+filename%3D%22rust_model.ot%22%3B&x-id=GetObject&Expires=1750779722&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1MDc3OTcyMn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82MjFmZmRjMTM2NDY4ZDcwOWYxODAyOTIvYzcwM2YwNmJlYmM5MmRiNGNlMzVkZmEwYmJlYTdjM2Y2ZThiNGI0YWNjZTYzNWI5ODJjYzg3MjdiOGU1NzNlZioifV19&Signature=HVeHQ2hnpFJeUy7Dc4ZkljDtGa1OcN1z0jBB9WVE3oTguch6OzLs3dRryz%7EyKESoaR0n4WG1qjH8xiLJZf9sAXlyJcvTgE9ZK7x3Sly94ZvFqSqBfILZUfAdcdAVpaBahjWMTfyTAL-jNhZ6CA1IgSUI8lnEW%7ExxruFzxkockwLKqEswcEFdfGqEKQwZ4X4GMJa-FHdKhgZeHOwzgOM4aQCkL8Ikbyx4qszqK3O3OYLwac2ASevLkjYno3UR3XN--kDTuIhxf46fbcPfAByTscRBfhBB4ZJ%7EeH2Mge1YePiYA6yA4pIeRY0PCfWsAzl%7ETNb5bQJQm4PNDeObvtjllA__&Key-Pair-Id=K2L8F4GPSG1IFC
`Rename the last one to sentenceMini.ot so it doesn´t conflict with rust_model.ot`
## Building the Core Library
```bash
cd core
cargo run
```

## Running the Mobile App
```bash
cd mobile
npm install
npm run dev
```
# idea-capture
