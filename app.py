"""
Dublador de vídeos: inglês -> português (Brasil)

Fluxo:
  1. Baixa o vídeo do link (YouTube e outros sites) com yt-dlp
  2. Transcreve o áudio em inglês com faster-whisper (roda local)
  3. Traduz cada frase para português (Google Tradutor, gratuito)
  4. Gera a voz em português com edge-tts (vozes neurais gratuitas)
  5. Encaixa cada frase no tempo certo e junta com o vídeo (ffmpeg)

Uso: python app.py  e abra http://localhost:7860
"""

import asyncio
import shutil
import subprocess
import uuid
import wave
from pathlib import Path

import edge_tts
import gradio as gr
import numpy as np
import yt_dlp
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel

SR = 24000  # taxa de amostragem do áudio dublado
PASTA_SAIDA = Path("saidas")
PASTA_SAIDA.mkdir(exist_ok=True)

VOZES = {
    "Antonio (masculina)": "pt-BR-AntonioNeural",
    "Francisca (feminina)": "pt-BR-FranciscaNeural",
    "Thalita (feminina)": "pt-BR-ThalitaNeural",
}

_modelos = {}


# ---------- utilidades ----------

def rodar(cmd):
    return subprocess.run(cmd, check=True, capture_output=True)


def duracao(arquivo):
    out = rodar([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(arquivo),
    ]).stdout.decode().strip()
    return float(out)


def tem_audio(arquivo):
    out = rodar([
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(arquivo),
    ]).stdout.decode().strip()
    return bool(out)


def carregar_pcm(arquivo, tempo=1.0):
    """Lê um áudio como array float mono, opcionalmente acelerado."""
    filtro = ["-filter:a", f"atempo={tempo:.3f}"] if tempo > 1.0 else []
    raw = rodar([
        "ffmpeg", "-v", "error", "-i", str(arquivo), *filtro,
        "-f", "s16le", "-ac", "1", "-ar", str(SR), "-",
    ]).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def salvar_wav(arquivo, dados):
    pcm = (np.clip(dados, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(arquivo), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def formato_srt(seg):
    h, resto = divmod(seg, 3600)
    m, s = divmod(resto, 60)
    ms = int(round((s - int(s)) * 1000))
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"


# ---------- etapas ----------

def baixar_video(url, pasta):
    opcoes = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "outtmpl": str(pasta / "original.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(opcoes) as ydl:
        ydl.download([url])
    arquivos = sorted(pasta.glob("original.*"))
    if not arquivos:
        raise RuntimeError("Não foi possível baixar o vídeo desse link.")
    return arquivos[0]


def transcrever(video, pasta, tamanho_modelo):
    audio = pasta / "audio_en.wav"
    rodar(["ffmpeg", "-v", "error", "-y", "-i", str(video),
           "-ac", "1", "-ar", "16000", str(audio)])
    if tamanho_modelo not in _modelos:
        _modelos[tamanho_modelo] = WhisperModel(
            tamanho_modelo, device="auto", compute_type="default")
    segmentos, _ = _modelos[tamanho_modelo].transcribe(
        str(audio), language="en", vad_filter=True)
    return [{"inicio": s.start, "fim": s.end, "en": s.text.strip()}
            for s in segmentos if s.text.strip()]


def traduzir(segmentos):
    tradutor = GoogleTranslator(source="en", target="pt")
    for s in segmentos:
        try:
            s["pt"] = tradutor.translate(s["en"]) or s["en"]
        except Exception:
            s["pt"] = s["en"]
    return segmentos


async def _gerar_vozes(segmentos, voz, pasta):
    for i, s in enumerate(segmentos):
        arq = pasta / f"fala_{i:04}.mp3"
        await edge_tts.Communicate(s["pt"], voz).save(str(arq))
        s["arquivo"] = arq


def montar_audio(segmentos, duracao_total):
    """Coloca cada fala no seu tempo, acelerando se não couber."""
    trilha = np.zeros(int((duracao_total + 5) * SR), dtype=np.float32)
    for i, s in enumerate(segmentos):
        clipe = carregar_pcm(s["arquivo"])
        proximo = segmentos[i + 1]["inicio"] if i + 1 < len(segmentos) else duracao_total
        espaco = max(proximo - s["inicio"], 0.3)
        dur = len(clipe) / SR
        if dur > espaco:
            clipe = carregar_pcm(s["arquivo"], tempo=min(dur / espaco, 1.6))
        ini = int(s["inicio"] * SR)
        fim = min(ini + len(clipe), len(trilha))
        trilha[ini:fim] += clipe[: fim - ini]
    return trilha[: int(duracao_total * SR)]


def juntar(video, audio_pt, saida, volume_original):
    if volume_original > 0 and tem_audio(video):
        filtro = (f"[0:a]volume={volume_original}[a0];"
                  "[a0][1:a]amix=inputs=2:duration=first:normalize=0[a]")
        mapa = ["-filter_complex", filtro, "-map", "0:v", "-map", "[a]"]
    else:
        mapa = ["-map", "0:v", "-map", "1:a"]
    rodar(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(audio_pt),
           *mapa, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", str(saida)])


def salvar_srt(segmentos, arquivo):
    linhas = []
    for i, s in enumerate(segmentos, 1):
        linhas += [str(i), f"{formato_srt(s['inicio'])} --> {formato_srt(s['fim'])}",
                   s["pt"], ""]
    arquivo.write_text("\n".join(linhas), encoding="utf-8")


# ---------- função principal ----------

def dublar(url, arquivo_enviado, nome_voz, volume_original, tamanho_modelo,
           progresso=gr.Progress()):
    if not url and not arquivo_enviado:
        raise gr.Error("Cole um link ou envie um arquivo de vídeo.")

    pasta = PASTA_SAIDA / uuid.uuid4().hex[:8]
    pasta.mkdir(parents=True)
    temp = pasta / "temp"
    temp.mkdir()

    try:
        progresso(0.05, desc="Baixando vídeo...")
        if arquivo_enviado:
            video = temp / ("original" + Path(arquivo_enviado).suffix)
            shutil.copy(arquivo_enviado, video)
        else:
            video = baixar_video(url.strip(), temp)

        progresso(0.2, desc="Transcrevendo o inglês (pode demorar)...")
        segmentos = transcrever(video, temp, tamanho_modelo)
        if not segmentos:
            raise gr.Error("Não encontrei fala em inglês nesse vídeo.")

        progresso(0.5, desc="Traduzindo para português...")
        traduzir(segmentos)

        progresso(0.65, desc="Gerando a voz em português...")
        asyncio.run(_gerar_vozes(segmentos, VOZES[nome_voz], temp))

        progresso(0.85, desc="Montando o vídeo dublado...")
        trilha = montar_audio(segmentos, duracao(video))
        audio_pt = temp / "audio_pt.wav"
        salvar_wav(audio_pt, trilha)

        saida = pasta / "video_dublado.mp4"
        juntar(video, audio_pt, saida, volume_original)
        srt = pasta / "legendas_pt.srt"
        salvar_srt(segmentos, srt)
    except gr.Error:
        raise
    except subprocess.CalledProcessError as e:
        raise gr.Error("Erro no ffmpeg: " + e.stderr.decode(errors="ignore")[-300:])
    except Exception as e:
        raise gr.Error(f"Erro: {e}")
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    progresso(1.0, desc="Pronto!")
    return str(saida), str(srt)


# ---------- interface ----------

with gr.Blocks(title="Dublador EN para PT") as app:
    gr.Markdown("# Dublador de vídeos: inglês para português\n"
                "Cole o link de um vídeo em inglês (YouTube, Vimeo etc.) "
                "e clique em **Dublar**.")
    with gr.Row():
        with gr.Column():
            url = gr.Textbox(label="Link do vídeo",
                             placeholder="https://www.youtube.com/watch?v=...")
            arquivo = gr.File(label="Ou envie um arquivo de vídeo",
                              file_types=["video"], type="filepath")
            voz = gr.Dropdown(list(VOZES), value="Antonio (masculina)", label="Voz")
            volume = gr.Slider(0, 0.5, value=0.1, step=0.05,
                               label="Volume do áudio original ao fundo")
            modelo = gr.Dropdown(["tiny", "base", "small", "medium"], value="small",
                                 label="Precisão da transcrição (maior = melhor e mais lento)")
            botao = gr.Button("Dublar", variant="primary")
        with gr.Column():
            video_saida = gr.Video(label="Vídeo dublado")
            srt_saida = gr.File(label="Legendas em português (.srt)")

    botao.click(dublar, [url, arquivo, voz, volume, modelo], [video_saida, srt_saida])

if __name__ == "__main__":
    app.launch(inbrowser=True)
