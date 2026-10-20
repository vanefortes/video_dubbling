# Dublador de vídeos: inglês para português

Cole o link de um vídeo em inglês e receba o vídeo dublado em português do Brasil, mais as legendas em .srt. Tudo gratuito.

## O que você precisa

1. **Python 3.10 ou mais novo**: https://www.python.org/downloads/ (no Windows, marque "Add Python to PATH" na instalação)
2. **FFmpeg**:
   * Windows: abra o PowerShell e rode `winget install ffmpeg` (depois feche e abra o terminal de novo)
   * Mac: `brew install ffmpeg`
   * Linux: `sudo apt install ffmpeg`

## Instalação (uma vez só)

Coloque `app.py` e `requirements.txt` numa pasta, abra o terminal nessa pasta e rode:

```
pip install -r requirements.txt
```

## Como usar

```
python app.py
```

O navegador abre em http://localhost:7860. Cole o link, escolha a voz e clique em **Dublar**.
O resultado também fica salvo na pasta `saidas/`.

## Dicas

* A primeira execução demora mais porque baixa o modelo de transcrição.
* Sem placa de vídeo, um vídeo de 10 minutos leva alguns minutos com o modelo "small". Use "base" ou "tiny" para ir mais rápido.
* Se o YouTube der erro ao baixar, atualize o yt-dlp: `pip install -U yt-dlp`
* Todas as pessoas do vídeo são dubladas com a mesma voz, e não há sincronia labial.
* Quando a fala em português é mais longa que a original, ela é acelerada um pouco para caber no tempo.
