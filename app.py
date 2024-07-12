import torch
import whisper
from flask import Flask, request, jsonify, render_template
import logging
import time
import ffmpeg

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verifique se a GPU está disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Carregue o modelo Whisper
logger.info("Loading Whisper model...")
model = whisper.load_model("medium", device=device)
logger.info("Model loaded successfully.")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def get_audio_duration(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        return None

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        logger.warning("No audio file provided in the request.")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_path = "uploaded_audio.mp3"
    audio_file.save(audio_path)
    logger.info(f"Audio file saved to {audio_path}")

    # Obtém a duração do áudio
    audio_duration = get_audio_duration(audio_path)
    if audio_duration is None:
        return jsonify({"error": "Could not determine audio duration"}), 500

    # Transcreva o áudio
    logger.info("Starting transcription...")
    start_time = time.time()
    result = model.transcribe(audio_path, language="pt", word_timestamps=True)
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Transcription completed in {duration:.2f} seconds.")

    # Retorne a transcrição, o tempo gasto e a duração do áudio
    return jsonify({
        "transcription": result["text"],
        "duration": duration,
        "audio_duration": audio_duration,
        "word_timestamps": result["segments"]
    })

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
