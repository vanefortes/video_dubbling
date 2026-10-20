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
import threading
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

MODOS = ["Dublagem e legenda", "Só dublagem", "Só legenda"]
ETAPAS = ["Download", "Transcrição", "Tradução", "Geração da voz", "Montagem do vídeo"]

# Ritmo de cada etapa, usado para prever o tempo das etapas que ainda não
# começaram: segundos por segundo de vídeo (transcrição, por modelo) ou
# segundos por frase (demais). É atualizado com o que foi medido na última dublagem.
_ritmo = {"tiny": 0.05, "base": 0.1, "small": 0.3, "medium": 0.8,
          "Tradução": 0.5, "Geração da voz": 1.0, "Montagem do vídeo": 0.05}
SEGUNDOS_POR_FRASE = 4  # para estimar quantas frases o vídeo tem antes de transcrever

_modelos = {}
_parar = set()  # sessões em que clicaram em "Parar"
_logs = {}  # log de cada sessão, mostrado na tela
_local = threading.local()  # log da dublagem que roda nesta thread


# ---------- utilidades ----------

def registrar(msg):
    linha = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(linha)
    log = getattr(_local, "log", None)
    if log is not None:
        log.append(linha)


def posicao(seg):
    return f"{int(seg // 60):02}:{int(seg % 60):02}"


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
    return (f"⏱️ Concluído em **{formato_tempo(total)}** "
            f"(vídeo de {formato_tempo(duracao_video)})\n\n" + "\n".join(linhas))


class Relogio:
    """Mede as etapas da dublagem e mostra quanto tempo falta."""

    def __init__(self, progresso, modelo, etapas):
        self.progresso = progresso
        self.modelo = modelo
        self.etapas = etapas
        self.marcas = []  # (nome da etapa, horário em que começou)
        self.duracao_video = 0
        self.frases = 0
        self._ultima_tela = 0

    def etapa(self, nome, desc):
        self._guardar_ritmo()
        self.marcas.append((nome, time.monotonic()))
        self.desc = desc
        self._ultima_tela = 0
        self.avancar(0, 1)

    def fim(self):
        self._guardar_ritmo()
        self.marcas.append(("Fim", time.monotonic()))

    def avancar(self, feito, total, falta_etapa=None):
        agora = time.monotonic()
        if agora - self._ultima_tela < 0.5:
            return
        self._ultima_tela = agora
        if not self.duracao_video:  # ainda não sabe o tamanho do vídeo
            self.progresso(0.02, desc=f"{self.desc} Calculando o tempo...")
            return

        nome, inicio = self.marcas[-1]
        if falta_etapa is None:
            if feito > 0:  # proporcional ao que já foi feito nesta etapa
                falta_etapa = (agora - inicio) * (total - feito) / feito
            else:
                falta_etapa = self._previsto(nome)
        seguintes = self.etapas[self.etapas.index(nome) + 1:]
        falta = max(falta_etapa, 0) + sum(self._previsto(e) for e in seguintes)
        decorrido = agora - self.marcas[0][1]
        self.progresso(decorrido / (decorrido + falta + 0.01),
                       desc=f"{self.desc} Faltam ~{formato_tempo(falta)}")

    def _chave(self, nome):
        return self.modelo if nome == "Transcrição" else nome

    def _quantidade(self, nome):
        """Tamanho do trabalho da etapa: segundos de vídeo ou número de frases."""
        if nome == "Transcrição":
            return self.duracao_video
        return self.frases or self.duracao_video / SEGUNDOS_POR_FRASE

    def _previsto(self, nome):
        return _ritmo.get(self._chave(nome), 0) * self._quantidade(nome)

    def _guardar_ritmo(self):
        if not self.marcas or self.marcas[-1][0] == "Download":
            return
        nome, inicio = self.marcas[-1]
        if self._quantidade(nome):
            _ritmo[self._chave(nome)] = (time.monotonic() - inicio) / self._quantidade(nome)


# ---------- etapas ----------

def baixar_video(url, pasta, gancho):
    opcoes = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "outtmpl": str(pasta / "original.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "progress_hooks": [gancho],
    }
    with yt_dlp.YoutubeDL(opcoes) as ydl:
        ydl.download([url])
    arquivos = sorted(pasta.glob("original.*"))
    if not arquivos:
        raise RuntimeError("Não foi possível baixar o vídeo desse link.")
    return arquivos[0]


def transcrever(video, pasta, tamanho_modelo, duracao_video, passo):
    audio = pasta / "audio_en.wav"
    rodar(["ffmpeg", "-v", "error", "-y", "-i", str(video),
           "-ac", "1", "-ar", "16000", str(audio)])
    if tamanho_modelo not in _modelos:
        registrar(f"Carregando o modelo de transcrição '{tamanho_modelo}' "
                  "(na primeira vez ele é baixado)...")
        _modelos[tamanho_modelo] = WhisperModel(
            tamanho_modelo, device="auto", compute_type="default")
    try:
        return _segmentos(_modelos[tamanho_modelo], audio, duracao_video, passo)
    except RuntimeError as e:
        # placa NVIDIA sem as bibliotecas CUDA: usa o processador
        if not any(p in str(e).lower() for p in ("cuda", "cublas", "cudnn")):
            raise
        registrar("Placa de vídeo indisponível (faltam as bibliotecas CUDA). "
                  "Transcrevendo com o processador.")
        _modelos[tamanho_modelo] = WhisperModel(
            tamanho_modelo, device="cpu", compute_type="int8")
        return _segmentos(_modelos[tamanho_modelo], audio, duracao_video, passo)


def _segmentos(modelo, audio, duracao_video, passo):
    segmentos, _ = modelo.transcribe(str(audio), language="en", vad_filter=True)
    resultado = []
    for s in segmentos:  # o whisper transcreve uma frase por vez
        passo(s.end, duracao_video)
        if s.text.strip():
            registrar(f"[{posicao(s.start)}] EN: {s.text.strip()}")
            resultado.append({"inicio": s.start, "fim": s.end, "en": s.text.strip()})
    return resultado


def traduzir(segmentos, passo):
    tradutores = [GoogleTranslator(source="en", target="pt"),
                  MyMemoryTranslator(source="en-US", target="pt-BR")]
    for i, s in enumerate(segmentos):
        passo(i, len(segmentos))
        while tradutores:
            try:
                s["pt"] = tradutores[0].translate(s["en"]) or s["en"]
                registrar(f"[{posicao(s['inicio'])}] PT: {s['pt']}")
                break
            except Exception as e:
                # recusou (ex.: Google bloqueou): usa o próximo
                registrar(f"{type(tradutores.pop(0)).__name__} recusou "
                          f"({type(e).__name__}). Tentando o próximo tradutor.")
        else:
            raise gr.Error("Os tradutores gratuitos recusaram o pedido. "
                           "Tente de novo mais tarde.")
    return segmentos


async def _gerar_vozes(segmentos, voz, pasta, passo):
    for i, s in enumerate(segmentos):
        passo(i, len(segmentos))
        arq = pasta / f"fala_{i:04}.mp3"
        await edge_tts.Communicate(s["pt"], voz).save(str(arq))
        s["arquivo"] = arq
        registrar(f"Voz gerada: frase {i + 1} de {len(segmentos)}")


def montar_audio(segmentos, duracao_total, passo):
    """Coloca cada fala no seu tempo, acelerando se não couber."""
    trilha = np.zeros(int((duracao_total + 5) * SR), dtype=np.float32)
    for i, s in enumerate(segmentos):
        passo(i, len(segmentos))
        clipe = carregar_pcm(s["arquivo"])
        proximo = segmentos[i + 1]["inicio"] if i + 1 < len(segmentos) else duracao_total
        espaco = max(proximo - s["inicio"], 0.3)
        dur = len(clipe) / SR
        if dur > espaco:
            tempo = min(dur / espaco, 1.6)
            registrar(f"[{posicao(s['inicio'])}] Fala em português mais longa: "
                      f"acelerada {tempo:.2f}x")
            clipe = carregar_pcm(s["arquivo"], tempo=tempo)
        ini = int(s["inicio"] * SR)
        fim = min(ini + len(clipe), len(trilha))
        trilha[ini:fim] += clipe[: fim - ini]
    return trilha[: int(duracao_total * SR)]


def juntar(video, saida, audio_pt=None, volume_original=0, srt=None):
    """Monta o vídeo final com a voz em português e/ou a faixa de legenda."""
    entradas = ["-i", str(video)]
    if audio_pt:
        entradas += ["-i", str(audio_pt)]
        if volume_original > 0 and tem_audio(video):
            filtro = (f"[0:a]volume={volume_original}[a0];"
                      "[a0][1:a]amix=inputs=2:duration=first:normalize=0[a]")
            mapa = ["-filter_complex", filtro, "-map", "0:v", "-map", "[a]"]
        else:
            mapa = ["-map", "0:v", "-map", "1:a"]
        mapa += ["-shortest"]
    else:  # só legenda: mantém o áudio original
        mapa = ["-map", "0:v", "-map", "0:a?"]
    if srt:
        mapa += ["-map", f"{len(entradas) // 2}:s", "-c:s", "mov_text",
                 "-metadata:s:s:0", "language=por"]
        entradas += ["-i", str(srt)]
    rodar(["ffmpeg", "-v", "error", "-y", *entradas, *mapa,
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(saida)])


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


def mostrar_log(request: gr.Request):
    return "\n".join(_logs.get(request.session_hash, [])[-500:])


def dublar(url, arquivo_enviado, modo, nome_voz, volume_original, tamanho_modelo,
           request: gr.Request, progresso=gr.Progress()):
    if not url and not arquivo_enviado:
        raise gr.Error("Cole um link ou envie um arquivo de vídeo.")

    dublagem = modo != "Só legenda"
    legenda = modo != "Só dublagem"

    sessao = request.session_hash
    _parar.discard(sessao)
    _logs[sessao] = _local.log = []
    registrar(f"Início: {'arquivo enviado' if arquivo_enviado else url.strip()} "
              f"| {modo}" + (f" | voz {nome_voz}" if dublagem else "")
              + f" | precisão '{tamanho_modelo}'")

    def checar():
        if sessao in _parar:
            raise gr.Error("Dublagem interrompida.")

    pasta = PASTA_SAIDA / uuid.uuid4().hex[:8]
    pasta.mkdir(parents=True)
    temp = pasta / "temp"
    temp.mkdir()

    relogio = Relogio(progresso, tamanho_modelo,
                      ETAPAS if dublagem else [e for e in ETAPAS if e != "Geração da voz"])

    def etapa(nome, desc):
        checar()
        relogio.etapa(nome, desc)

    def passo(feito, total, falta_etapa=None):
        checar()
        relogio.avancar(feito, total, falta_etapa)

    def gancho_download(d):
        if d["status"] == "finished":
            registrar(f"Arquivo baixado: {Path(d['filename']).name}")
        if d["status"] == "downloading":
            relogio.duracao_video = d.get("info_dict", {}).get("duration") or 0
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            passo(d.get("downloaded_bytes") or 0, total or 1, d.get("eta"))

    try:
        etapa("Download", "Baixando vídeo...")
        if arquivo_enviado:
            video = temp / ("original" + Path(arquivo_enviado).suffix)
            shutil.copy(arquivo_enviado, video)
        else:
            video = baixar_video(url.strip(), temp, gancho_download)
        duracao_video = relogio.duracao_video = duracao(video)
        registrar(f"Vídeo pronto: {formato_tempo(duracao_video)} de duração")

        etapa("Transcrição", "Transcrevendo o inglês...")
        segmentos = transcrever(video, temp, tamanho_modelo, duracao_video, passo)
        if not segmentos:
            raise gr.Error("Não encontrei fala em inglês nesse vídeo.")
        relogio.frases = len(segmentos)
        registrar(f"Transcrição concluída: {len(segmentos)} frases")

        etapa("Tradução", "Traduzindo para português...")
        traduzir(segmentos, passo)

        srt = pasta / "legendas_pt.srt" if legenda else None
        if legenda:
            salvar_srt(segmentos, srt)

        audio_pt = None
        if dublagem:
            etapa("Geração da voz", "Gerando a voz em português...")
            asyncio.run(_gerar_vozes(segmentos, VOZES[nome_voz], temp, passo))

            etapa("Montagem do vídeo", "Montando o vídeo dublado...")
            trilha = montar_audio(segmentos, duracao_video, passo)
            audio_pt = temp / "audio_pt.wav"
            salvar_wav(audio_pt, trilha)
        else:
            etapa("Montagem do vídeo", "Colocando a legenda no vídeo...")

        registrar("Montando o vídeo final"
                  + (" com a voz em português" if dublagem else "")
                  + (" e a faixa de legenda" if dublagem and legenda else "")
                  + (" com a faixa de legenda" if legenda and not dublagem else "")
                  + "...")
        saida = pasta / ("video_dublado.mp4" if dublagem else "video_legendado.mp4")
        juntar(video, saida, audio_pt, volume_original, srt)
        relogio.fim()
        registrar(f"Pronto! Arquivos salvos em {pasta}")
    except gr.Error as e:
        registrar(e.message)
        raise
    except subprocess.CalledProcessError as e:
        registrar("Erro no ffmpeg: " + e.stderr.decode(errors="ignore")[-300:])
        raise gr.Error("Erro no ffmpeg: " + e.stderr.decode(errors="ignore")[-300:])
    except Exception as e:
        if sessao in _parar:  # o yt-dlp embrulha o erro do "Parar" em outro erro
            registrar("Dublagem interrompida.")
            raise gr.Error("Dublagem interrompida.")
        registrar(f"Erro: {e}")
        raise gr.Error(f"Erro: {e}")
    finally:
        _parar.discard(sessao)
        _local.log = None
        shutil.rmtree(temp, ignore_errors=True)

    progresso(1.0, desc="Pronto!")
    resumo = resumo_tempos(relogio.marcas, duracao_video)
    print(resumo.replace("**", ""))
    return (gr.Video(str(saida), subtitles=str(srt) if srt else None,
                     label="Vídeo dublado" if dublagem else "Vídeo com legenda"),
            gr.File(str(srt) if srt else None, visible=legenda),
            resumo)


def fim_do_log(request: gr.Request):
    return mostrar_log(request), gr.Timer(active=False)


def trocar_modo(modo):
    dublagem = modo != "Só legenda"
    return (gr.Column(visible=dublagem),
            gr.Button("Dublar" if dublagem else "Gerar legenda"))


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
            modo = gr.Radio(MODOS, value=MODOS[0], label="O que você quer gerar?")
            with gr.Column() as opcoes_voz:
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
    log_saida = gr.Textbox(label="Log do processo", lines=12, max_lines=12,
                           autoscroll=True, interactive=False)
    relogio_log = gr.Timer(1, active=False)

    modo.change(trocar_modo, modo, [opcoes_voz, botao])
    botao_ouvir.click(lambda nome: gr.Audio(ouvir_voz(nome), visible=True),
                      voz, amostra)
    relogio_log.tick(mostrar_log, None, log_saida, show_progress="hidden")
    botao.click(lambda: gr.Timer(active=True), None, relogio_log).then(
        dublar, [url, arquivo, modo, voz, volume, modelo],
        [video_saida, srt_saida, tempo_saida],
        show_progress_on=[video_saida]).then(
        fim_do_log, None, [log_saida, relogio_log])
    botao_parar.click(parar, queue=False)

if __name__ == "__main__":
    app.launch(inbrowser=True, favicon_path=str(Path(__file__).parent / "icone.png"))
