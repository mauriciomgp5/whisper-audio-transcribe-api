import torch
import whisper
from flask import Flask, request, jsonify

# Verifique se a GPU está disponível
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carregue o modelo Whisper
model = whisper.load_model("base", device=device)

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_path = "uploaded_audio.mp3"
    audio_file.save(audio_path)

    # Transcreva o áudio
    result = model.transcribe(audio_path)

    # Retorne a transcrição
    return jsonify({"transcription": result["text"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
