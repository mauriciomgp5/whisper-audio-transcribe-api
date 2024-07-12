# Use uma imagem base com suporte a CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Configure o ambiente
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Use python3.9 e pip3.9 como padrão
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Instale o PyTorch com suporte a CUDA
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instale o Whisper, Flask, Gunicorn e ffmpeg-python
RUN pip3 install git+https://github.com/openai/whisper.git
RUN pip3 install Flask gunicorn ffmpeg-python

# Baixe o modelo Whisper durante a construção do contêiner
RUN python3 -c "import whisper; whisper.load_model('medium')"

# Defina o diretório de trabalho
WORKDIR /workspace

# Copie o script da API e templates para o contêiner
COPY app.py .
COPY templates ./templates

# Comando para executar o script com Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "app:app"]
