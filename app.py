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
import time
import uuid
import wave
from pathlib import Path

import edge_tts
import gradio as gr
import numpy as np
import yt_dlp
from deep_translator import GoogleTranslator, MyMemoryTranslator
from faster_whisper import WhisperModel

SR = 24000  # taxa de amostragem do áudio dublado
PASTA_SAIDA = Path("saidas")
PASTA_SAIDA.mkdir(exist_ok=True)

VOZES = {
    "Antonio (masculina)": "pt-BR-AntonioNeural",
    "Francisca (feminina)": "pt-BR-FranciscaNeural",
    "Thalita (feminina)": "pt-BR-ThalitaNeural",
}

AMOSTRA = "Olá! Esta é a voz que vai dublar o seu vídeo em português."

_modelos = {}
_parar = set()  # sessões em que clicaram em "Parar"


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


def formato_tempo(seg):
    m, s = divmod(int(round(seg)), 60)
    return f"{m} min {s:02} s" if m else f"{s} s"


def resumo_tempos(marcas, duracao_video):
    """Monta o texto com o tempo de cada etapa e o total."""
    linhas = ["| Etapa | Tempo |", "|---|---|"]
    for (nome, ini), (_, fim) in zip(marcas, marcas[1:]):
        linhas.append(f"| {nome} | {formato_tempo(fim - ini)} |")
    total = marcas[-1][1] - marcas[0][1]
    linhas.append(f"| **Total** | **{formato_tempo(total)}** |")
    return (f"⏱️ Dublagem concluída em **{formato_tempo(total)}** "
            f"(vídeo de {formato_tempo(duracao_video)})\n\n" + "\n".join(linhas))


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


def transcrever(video, pasta, tamanho_modelo, checar):
    audio = pasta / "audio_en.wav"
    rodar(["ffmpeg", "-v", "error", "-y", "-i", str(video),
           "-ac", "1", "-ar", "16000", str(audio)])
    if tamanho_modelo not in _modelos:
        _modelos[tamanho_modelo] = WhisperModel(
            tamanho_modelo, device="auto", compute_type="default")
    try:
        return _segmentos(_modelos[tamanho_modelo], audio, checar)
    except RuntimeError as e:
        # placa NVIDIA sem as bibliotecas CUDA: usa o processador
        if not any(p in str(e).lower() for p in ("cuda", "cublas", "cudnn")):
            raise
        _modelos[tamanho_modelo] = WhisperModel(
            tamanho_modelo, device="cpu", compute_type="int8")
        return _segmentos(_modelos[tamanho_modelo], audio, checar)


def _segmentos(modelo, audio, checar):
    segmentos, _ = modelo.transcribe(str(audio), language="en", vad_filter=True)
    resultado = []
    for s in segmentos:  # o whisper transcreve uma frase por vez
        checar()
        if s.text.strip():
            resultado.append({"inicio": s.start, "fim": s.end, "en": s.text.strip()})
    return resultado


def traduzir(segmentos, checar):
    tradutores = [GoogleTranslator(source="en", target="pt"),
                  MyMemoryTranslator(source="en-US", target="pt-BR")]
    for s in segmentos:
        checar()
        while tradutores:
            try:
                s["pt"] = tradutores[0].translate(s["en"]) or s["en"]
                break
            except Exception:
                tradutores.pop(0)  # recusou (ex.: Google bloqueou): usa o próximo
        else:
            raise gr.Error("Os tradutores gratuitos recusaram o pedido. "
                           "Tente de novo mais tarde.")
    return segmentos


async def _gerar_vozes(segmentos, voz, pasta, checar):
    for i, s in enumerate(segmentos):
        checar()
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

def ouvir_voz(nome_voz):
    pasta = PASTA_SAIDA / "amostras"
    pasta.mkdir(exist_ok=True)
    arq = pasta / f"{VOZES[nome_voz]}.mp3"
    if not arq.exists():
        asyncio.run(edge_tts.Communicate(AMOSTRA, VOZES[nome_voz]).save(str(arq)))
    return str(arq)


def parar(request: gr.Request):
    _parar.add(request.session_hash)
    gr.Info("Parando... a etapa atual termina em instantes.")


def dublar(url, arquivo_enviado, nome_voz, volume_original, tamanho_modelo,
           request: gr.Request, progresso=gr.Progress()):
    if not url and not arquivo_enviado:
        raise gr.Error("Cole um link ou envie um arquivo de vídeo.")

    sessao = request.session_hash
    _parar.discard(sessao)

    def checar():
        if sessao in _parar:
            raise gr.Error("Dublagem interrompida.")

    pasta = PASTA_SAIDA / uuid.uuid4().hex[:8]
    pasta.mkdir(parents=True)
    temp = pasta / "temp"
    temp.mkdir()

    marcas = []  # (nome da etapa, horário em que começou)

    def etapa(fracao, desc, nome):
        checar()
        marcas.append((nome, time.monotonic()))
        progresso(fracao, desc=desc)

    try:
        etapa(0.05, "Baixando vídeo...", "Download")
        if arquivo_enviado:
            video = temp / ("original" + Path(arquivo_enviado).suffix)
            shutil.copy(arquivo_enviado, video)
        else:
            video = baixar_video(url.strip(), temp)

        etapa(0.2, "Transcrevendo o inglês (pode demorar)...", "Transcrição")
        segmentos = transcrever(video, temp, tamanho_modelo, checar)
        if not segmentos:
            raise gr.Error("Não encontrei fala em inglês nesse vídeo.")

        etapa(0.5, "Traduzindo para português...", "Tradução")
        traduzir(segmentos, checar)

        etapa(0.65, "Gerando a voz em português...", "Geração da voz")
        asyncio.run(_gerar_vozes(segmentos, VOZES[nome_voz], temp, checar))

        etapa(0.85, "Montando o vídeo dublado...", "Montagem do vídeo")
        duracao_video = duracao(video)
        trilha = montar_audio(segmentos, duracao_video)
        audio_pt = temp / "audio_pt.wav"
        salvar_wav(audio_pt, trilha)

        saida = pasta / "video_dublado.mp4"
        juntar(video, audio_pt, saida, volume_original)
        srt = pasta / "legendas_pt.srt"
        salvar_srt(segmentos, srt)
        marcas.append(("Fim", time.monotonic()))
    except gr.Error:
        raise
    except subprocess.CalledProcessError as e:
        raise gr.Error("Erro no ffmpeg: " + e.stderr.decode(errors="ignore")[-300:])
    except Exception as e:
        raise gr.Error(f"Erro: {e}")
    finally:
        _parar.discard(sessao)
        shutil.rmtree(temp, ignore_errors=True)

    progresso(1.0, desc="Pronto!")
    resumo = resumo_tempos(marcas, duracao_video)
    print(resumo.replace("**", ""))
    return str(saida), str(srt), resumo


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
            botao_ouvir = gr.Button("🔊 Ouvir a voz", size="sm")
            amostra = gr.Audio(label="Amostra da voz", autoplay=True,
                               interactive=False, visible=False)
            volume = gr.Slider(0, 0.5, value=0.1, step=0.05,
                               label="Volume do áudio original ao fundo")
            modelo = gr.Dropdown(["tiny", "base", "small", "medium"], value="small",
                                 label="Precisão da transcrição (maior = melhor e mais lento)")
            with gr.Row():
                botao = gr.Button("Dublar", variant="primary")
                botao_parar = gr.Button("Parar", variant="stop")
        with gr.Column():
            video_saida = gr.Video(label="Vídeo dublado")
            srt_saida = gr.File(label="Legendas em português (.srt)")
            tempo_saida = gr.Markdown()

    botao_ouvir.click(lambda nome: gr.Audio(ouvir_voz(nome), visible=True),
                      voz, amostra)
    botao.click(dublar, [url, arquivo, voz, volume, modelo],
                [video_saida, srt_saida, tempo_saida],
                show_progress_on=[video_saida])
    botao_parar.click(parar, queue=False)

if __name__ == "__main__":
    app.launch(inbrowser=True, favicon_path=str(Path(__file__).parent / "icone.png"))
