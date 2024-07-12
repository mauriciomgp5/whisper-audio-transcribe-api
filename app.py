import torch
import whisper
from flask import Flask, request, jsonify
import logging

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verifique se a GPU está disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Carregue o modelo Whisper
logger.info("Loading Whisper model...")
model = whisper.load_model("base", device=device)
logger.info("Model loaded successfully.")

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        logger.warning("No audio file provided in the request.")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_path = "uploaded_audio.mp3"
    audio_file.save(audio_path)
    logger.info(f"Audio file saved to {audio_path}")

    # Transcreva o áudio
    logger.info("Starting transcription...")
    result = model.transcribe(audio_path)
    logger.info("Transcription completed.")

    # Retorne a transcrição
    return jsonify({"transcription": result["text"]})

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
