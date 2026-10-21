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
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid
import wave
import webbrowser
import zipfile
from pathlib import Path

import ctranslate2
import edge_tts
import gradio as gr
import numpy as np
import sentencepiece
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
VOLUME_ORIGINAL = 0  # volume do áudio em inglês ao fundo da dublagem (0 = sem áudio original)

MODOS = ["Dublagem e legenda", "Só dublagem", "Só legenda"]
TRADUTORES = ["No computador (rápido e sem limite)", "Online (Google ou MyMemory)"]

# Tradutor que roda no computador: modelo OPUS-MT inglês -> português do Brasil
# (licença CC-BY 4.0), empacotado pelo projeto Argos Translate
MODELO_TRADUCAO = "https://argos-net.com/v1/translate-en_pb-1_9.argosmodel"
PASTA_MODELOS = Path.home() / ".cache" / "dublador"
CONFIG = Path.home() / ".config" / "dublador.json"  # lembra a última pasta escolhida
ETAPAS = ["Download", "Transcrição", "Tradução", "Geração da voz", "Montagem do vídeo"]

# Ritmo de cada etapa, usado para prever o tempo das etapas que ainda não
# começaram: segundos por segundo de vídeo (transcrição, por modelo) ou
# segundos por frase (demais). É atualizado com o que foi medido na última dublagem.
_ritmo = {"tiny": 0.05, "base": 0.1, "small": 0.3, "medium": 0.8,
          "Tradução": 0.5, "Geração da voz": 1.0, "Montagem do vídeo": 0.05}
SEGUNDOS_POR_FRASE = 4  # para estimar quantas frases o vídeo tem antes de transcrever

_modelos = {}
_tradutor_local = None
_parar = set()  # sessões em que clicaram em "Parar"
_rodando = set()  # sessões com dublagem em andamento
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


def nome_arquivo(texto):
    """Mantém o nome do vídeo, tirando só os caracteres que não valem em nome de arquivo."""
    limpo = "".join(" " if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in texto)
    return " ".join(limpo.split())[:120].strip(" .") or "video"


def pasta_padrao():
    try:
        return json.loads(CONFIG.read_text())["pasta_destino"]
    except (OSError, ValueError, KeyError):
        videos = Path.home() / "Videos"
        return str((videos if videos.is_dir() else Path.home()) / "Dublados")


def lembrar_pasta(pasta):
    try:
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps({"pasta_destino": str(pasta)}))
    except OSError:
        pass  # só deixa de lembrar; não atrapalha a dublagem


def escolher_pasta(atual):
    """Abre a janela de escolha de pasta do sistema (o app roda neste computador)."""
    inicio = Path(atual).expanduser()
    while not inicio.is_dir() and inicio != inicio.parent:
        inicio = inicio.parent
    try:
        r = subprocess.run(["zenity", "--file-selection", "--directory",
                            "--title=Onde salvar os vídeos?", f"--filename={inicio}/"],
                           capture_output=True, text=True)
    except FileNotFoundError:
        gr.Warning("Não encontrei a janela de pastas (zenity). Digite o caminho no campo.")
        return atual
    return r.stdout.strip() or atual  # vazio = cancelou


def copiar_para(destino, arquivos):
    """Copia os arquivos para a pasta escolhida sem sobrescrever nada.
    Os arquivos de um mesmo vídeo ficam com o mesmo nome base, ex.:
    "Aula 1_dublado.mp4" e "Aula 1_dublado.srt" (os players abrem a legenda sozinhos)."""
    base = arquivos[0].stem
    n = 1
    while any((destino / f"{base}{a.suffix}").exists() for a in arquivos):
        n += 1
        base = f"{arquivos[0].stem} ({n})"
    copias = []
    for a in arquivos:
        alvo = destino / f"{base}{a.suffix}"
        try:
            os.link(a, alvo)  # mesmo disco: não ocupa espaço a mais
        except OSError:
            shutil.copy2(a, alvo)
        copias.append(alvo)
    return copias


class Lote:
    """Vídeos enviados de uma vez, para calcular o tempo que falta no total."""

    def __init__(self, total):
        self.total = total
        self.atual = 0  # posição do vídeo em andamento
        self.tempos = []  # quanto demorou cada vídeo concluído


class Relogio:
    """Mede as etapas da dublagem e mostra quanto tempo falta."""

    def __init__(self, progresso, modelo, etapas, lote):
        self.progresso = progresso
        self.modelo = modelo
        self.etapas = etapas
        self.lote = lote
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
        lote = self.lote
        prefixo = f"Vídeo {lote.atual + 1} de {lote.total} · " if lote.total > 1 else ""
        if not self.duracao_video:  # ainda não sabe o tamanho do vídeo
            feito = sum(lote.tempos)
            media = feito / len(lote.tempos) if lote.tempos else 0
            self.progresso(feito / (feito + media * (lote.total - lote.atual) + 0.01),
                           desc=f"{prefixo}{self.desc} Calculando o tempo...")
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
        # vídeos que ainda vão começar: média dos concluídos ou previsão deste
        media = sum(lote.tempos) / len(lote.tempos) if lote.tempos else decorrido + falta
        falta += (lote.total - lote.atual - 1) * media
        feito = sum(lote.tempos) + decorrido
        self.progresso(feito / (feito + falta + 0.01),
                       desc=f"{prefixo}{self.desc} Faltam ~{formato_tempo(falta)}")

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
        info = ydl.extract_info(url, download=True)
    arquivos = sorted(pasta.glob("original.*"))
    if not arquivos:
        raise RuntimeError("Não foi possível baixar o vídeo desse link.")
    return arquivos[0], info.get("title") or "video"


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


def traduzir(segmentos, passo, online):
    pendentes = _traduzir_online(segmentos, passo) if online else segmentos
    if pendentes:
        _traduzir_local(pendentes, passo)
    return segmentos


def _traduzir_online(segmentos, passo):
    """Traduz pela internet. Devolve as frases que ficaram sem tradução."""
    tradutores = [GoogleTranslator(source="en", target="pt"),
                  MyMemoryTranslator(source="en-US", target="pt-BR")]
    for i, s in enumerate(segmentos):
        passo(i, len(segmentos))
        while tradutores:
            try:
                pt = tradutores[0].translate(s["en"]) or s["en"]
                if "MYMEMORY WARNING" in pt:
                    raise RuntimeError("limite diário atingido")
                s["pt"] = pt
                registrar(f"[{posicao(s['inicio'])}] PT: {s['pt']}")
                break
            except Exception as e:
                # recusou (ex.: Google bloqueou): usa o próximo
                registrar(f"{type(tradutores.pop(0)).__name__} recusou "
                          f"({type(e).__name__}). Tentando o próximo tradutor.")
        else:
            registrar("Os tradutores online recusaram. "
                      "Continuando com o tradutor do computador.")
            return segmentos[i:]
    return []


def carregar_tradutor_local():
    global _tradutor_local
    if _tradutor_local is None:
        pasta = PASTA_MODELOS / "translate-en_pb-1_9"
        if not pasta.exists():
            registrar("Baixando o modelo de tradução (66 MB, só na primeira vez)...")
            PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
            arquivo = PASTA_MODELOS / "en_pb.zip"
            # o servidor recusa o User-Agent padrão do Python
            pedido = urllib.request.Request(MODELO_TRADUCAO,
                                            headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(pedido) as resposta, open(arquivo, "wb") as f:
                shutil.copyfileobj(resposta, f)
            temp = PASTA_MODELOS / "extraindo"
            shutil.rmtree(temp, ignore_errors=True)
            with zipfile.ZipFile(arquivo) as z:
                z.extractall(temp)
            (temp / pasta.name).rename(pasta)  # só aparece quando está completo
            shutil.rmtree(temp)
            arquivo.unlink()
        _tradutor_local = (
            sentencepiece.SentencePieceProcessor(model_file=str(pasta / "sentencepiece.model")),
            ctranslate2.Translator(str(pasta / "model"), device="cpu"))
    return _tradutor_local


def _traduzir_local(segmentos, passo, lote=16):
    sp, modelo = carregar_tradutor_local()
    for i in range(0, len(segmentos), lote):
        passo(i, len(segmentos))
        grupo = segmentos[i:i + lote]
        resultados = modelo.translate_batch(
            [sp.encode(s["en"], out_type=str) for s in grupo], beam_size=4)
        for s, r in zip(grupo, resultados):
            s["pt"] = "".join(r.hypotheses[0]).replace("▁", " ").strip() or s["en"]
            registrar(f"[{posicao(s['inicio'])}] PT: {s['pt']}")


async def _gerar_vozes(segmentos, voz, pasta, passo, tentativas=3):
    """Gera a fala de cada frase. Uma frase que falhar fica sem voz, sem parar o vídeo."""
    for i, s in enumerate(segmentos):
        passo(i, len(segmentos))
        if not any(c.isalnum() for c in s["pt"]):  # ex.: "...", "♪"
            registrar(f"[{posicao(s['inicio'])}] Frase sem texto para falar, "
                      f"pulando: {s['pt']}")
            continue
        arq = pasta / f"fala_{i:04}.mp3"
        for tentativa in range(1, tentativas + 1):
            try:
                await edge_tts.Communicate(s["pt"], voz).save(str(arq))
                s["arquivo"] = arq
                registrar(f"Voz gerada: frase {i + 1} de {len(segmentos)}")
                break
            except Exception as e:
                if tentativa < tentativas:
                    registrar(f"Serviço de voz falhou na frase {i + 1} ({e}). "
                              f"Tentando de novo ({tentativa + 1} de {tentativas})...")
                    await asyncio.sleep(2 * tentativa)
                else:
                    registrar(f"[{posicao(s['inicio'])}] Não consegui gerar a voz desta "
                              f"frase, ela fica sem dublagem: {s['pt']}")
    if not any("arquivo" in s for s in segmentos):
        raise gr.Error("O serviço de voz não respondeu. Verifique a internet "
                       "e tente de novo.")


def montar_audio(segmentos, duracao_total, passo):
    """Coloca cada fala no seu tempo, acelerando se não couber."""
    trilha = np.zeros(int((duracao_total + 5) * SR), dtype=np.float32)
    for i, s in enumerate(segmentos):
        passo(i, len(segmentos))
        if "arquivo" not in s:  # frase que ficou sem voz
            continue
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
                      "[a0][1:a]amix=inputs=2:duration=longest:normalize=0[a]")
            mapa = ["-filter_complex", filtro, "-map", "0:v", "-map", "[a]"]
        else:
            mapa = ["-map", "0:v", "-map", "1:a"]
        # sem -shortest: a trilha em português já tem a duração exata do vídeo,
        # e o -shortest cortaria o vídeo no fim da última legenda
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


def processar_video(tipo, origem, modo, nome_voz, volume_original, tamanho_modelo,
                    tradutor, destino, relogio, checar):
    """Dubla e/ou legenda um vídeo (link ou arquivo) e devolve o que foi gerado."""
    dublagem = modo != "Só legenda"
    legenda = modo != "Só dublagem"

    pasta = PASTA_SAIDA / uuid.uuid4().hex[:8]
    pasta.mkdir(parents=True)
    temp = pasta / "temp"
    temp.mkdir()

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
        if tipo == "arquivo":
            video = temp / ("original" + Path(origem).suffix)
            shutil.copy(origem, video)
            titulo = Path(origem).stem
        else:
            video, titulo = baixar_video(origem, temp, gancho_download)
        nome = nome_arquivo(titulo)
        duracao_video = relogio.duracao_video = duracao(video)
        registrar(f"Vídeo pronto: \"{titulo}\", {formato_tempo(duracao_video)} de duração")

        etapa("Transcrição", "Transcrevendo o inglês...")
        segmentos = transcrever(video, temp, tamanho_modelo, duracao_video, passo)
        if not segmentos:
            raise gr.Error("Não encontrei fala em inglês nesse vídeo.")
        relogio.frases = len(segmentos)
        registrar(f"Transcrição concluída: {len(segmentos)} frases")

        etapa("Tradução", "Traduzindo para português...")
        traduzir(segmentos, passo, online=tradutor != TRADUTORES[0])

        base = f"{nome}_{'dublado' if dublagem else 'legendado'}"
        srt = pasta / f"{base}.srt" if legenda else None
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
        saida = pasta / f"{base}.mp4"
        juntar(video, saida, audio_pt, volume_original, srt)
        copias = copiar_para(destino, [saida, srt] if srt else [saida])
        relogio.fim()
        total = relogio.marcas[-1][1] - relogio.marcas[0][1]
        registrar(f"Pronto em {formato_tempo(total)}! Salvo em: "
                  + ", ".join(str(c) for c in copias))
    except gr.Error:
        raise
    except subprocess.CalledProcessError as e:
        raise gr.Error("Erro no ffmpeg: " + e.stderr.decode(errors="ignore")[-300:])
    except yt_dlp.utils.DownloadError as e:
        checar()  # o yt-dlp embrulha o erro do "Parar" em outro erro
        registrar(f"Detalhe do erro: {e}")
        raise gr.Error("Não foi possível baixar o vídeo desse link.")
    except Exception as e:
        checar()
        raise gr.Error(f"Erro: {e}")
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    return {"titulo": titulo, "video": saida, "srt": srt, "tempo": total,
            "resumo": resumo_tempos(relogio.marcas, duracao_video)}


def resumo_lote(linhas, tempo_total):
    concluidos = sum(1 for _, resultado, _ in linhas if resultado.startswith("✅"))
    tabela = ["| Vídeo | Resultado | Tempo |", "|---|---|---|"]
    tabela += [f"| {titulo} | {resultado} | {tempo} |" for titulo, resultado, tempo in linhas]
    return (f"⏱️ **{concluidos} de {len(linhas)} vídeos** concluídos em "
            f"**{formato_tempo(tempo_total)}**\n\n" + "\n".join(tabela))


def dublar(links, arquivos, modo, nome_voz, tamanho_modelo,
           tradutor, pasta_destino, request: gr.Request, progresso=gr.Progress()):
    fila = [("link", l.strip()) for l in (links or "").splitlines() if l.strip()]
    fila += [("arquivo", a) for a in (arquivos or [])]
    if not fila:
        raise gr.Error("Cole um link ou envie um arquivo de vídeo.")

    if not (pasta_destino or "").strip():
        raise gr.Error("Escolha a pasta onde salvar os vídeos.")
    destino = Path(pasta_destino.strip()).expanduser()
    try:
        destino.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise gr.Error(f"Não consegui usar a pasta {destino}: {e.strerror}")
    if not os.access(destino, os.W_OK):
        raise gr.Error(f"Sem permissão para salvar na pasta {destino}.")
    lembrar_pasta(destino)

    dublagem = modo != "Só legenda"
    sessao = request.session_hash
    _parar.discard(sessao)
    _logs[sessao] = _local.log = []
    registrar(f"Início: {len(fila)} vídeo(s) | {modo}"
              + (f" | voz {nome_voz}" if dublagem else "")
              + f" | precisão '{tamanho_modelo}' | tradutor {tradutor}"
              + f" | salvar em {destino}")

    def checar():
        if sessao in _parar:
            raise gr.Error("Dublagem interrompida.")

    etapas = ETAPAS if dublagem else [e for e in ETAPAS if e != "Geração da voz"]
    lote = Lote(len(fila))
    inicio = time.monotonic()
    prontos = []  # resultados de processar_video
    linhas = []  # (título, resultado, tempo) para o resumo
    erro = None
    _rodando.add(sessao)
    try:
        for i, (tipo, origem) in enumerate(fila):
            lote.atual = i
            rotulo = origem if tipo == "link" else Path(origem).name
            if len(fila) > 1:
                registrar(f"===== Vídeo {i + 1} de {len(fila)}: {rotulo} =====")
            try:
                r = processar_video(tipo, origem, modo, nome_voz, VOLUME_ORIGINAL,
                                    tamanho_modelo, tradutor, destino,
                                    Relogio(progresso, tamanho_modelo, etapas, lote), checar)
            except gr.Error as e:
                erro = e.message
                if sessao in _parar:
                    registrar("Dublagem interrompida.")
                    linhas.append((rotulo, "⏹️ Interrompido", "-"))
                    linhas += [(o if t == "link" else Path(o).name, "⏹️ Não processado", "-")
                               for t, o in fila[i + 1:]]
                    break
                registrar(f"❌ {erro}"
                          + (" Seguindo para o próximo vídeo." if i + 1 < len(fila) else ""))
                linhas.append((rotulo, f"❌ {erro}", "-"))
                continue
            lote.tempos.append(r["tempo"])
            prontos.append(r)
            linhas.append((r["titulo"], "✅ Pronto", formato_tempo(r["tempo"])))
    finally:
        _rodando.discard(sessao)
        _parar.discard(sessao)
        _local.log = None

    if not prontos:
        raise gr.Error(erro)

    progresso(1.0, desc="Pronto!")
    if len(fila) == 1:
        resumo = prontos[0]["resumo"]
    else:
        resumo = resumo_lote(linhas, time.monotonic() - inicio)
    print(resumo.replace("**", ""))

    ultimo = prontos[-1]
    arquivos_gerados = [str(p) for r in prontos for p in (r["video"], r["srt"]) if p]
    return (gr.Video(str(ultimo["video"]),
                     subtitles=str(ultimo["srt"]) if ultimo["srt"] else None,
                     label=ultimo["titulo"]),
            arquivos_gerados,
            resumo)


def fim_do_log(request: gr.Request):
    return mostrar_log(request), gr.Timer(active=False)


def limpar(request: gr.Request):
    """Apaga entradas e resultados da tela; as escolhas (modo, voz, pasta...) ficam."""
    if request.session_hash in _rodando:
        gr.Warning("Há uma dublagem em andamento. Clique em Parar antes de limpar.")
        return [gr.skip()] * 7
    _logs.pop(request.session_hash, None)
    return "", None, None, None, "", "", gr.Audio(None, visible=False)


def trocar_modo(modo):
    dublagem = modo != "Só legenda"
    return (gr.Column(visible=dublagem),
            gr.Button("Dublar" if dublagem else "Gerar legenda"))


# ---------- interface ----------

with gr.Blocks(title="Dublador EN para PT") as app:
    gr.Markdown("# Dublador de vídeos: inglês para português\n"
                "Cole os links de vídeos em inglês (YouTube, Vimeo etc.) ou envie "
                "os arquivos e clique em **Dublar**. Dá para enviar vários de uma vez.")
    with gr.Row():
        with gr.Column():
            url = gr.Textbox(label="Links dos vídeos (um por linha)", lines=3,
                             placeholder="https://www.youtube.com/watch?v=...\n"
                                         "https://www.youtube.com/watch?v=...")
            arquivo = gr.File(label="Ou envie arquivos de vídeo", file_count="multiple",
                              file_types=["video"], type="filepath")
            modo = gr.Radio(MODOS, value=MODOS[0], label="O que você quer gerar?")
            with gr.Column() as opcoes_voz:
                voz = gr.Dropdown(list(VOZES), value="Antonio (masculina)", label="Voz")
                botao_ouvir = gr.Button("🔊 Ouvir a voz", size="sm")
                amostra = gr.Audio(label="Amostra da voz", autoplay=True,
                                   interactive=False, visible=False)
            modelo = gr.Dropdown(["tiny", "base", "small", "medium"], value="small",
                                 label="Precisão da transcrição (maior = melhor e mais lento)")
            tradutor = gr.Radio(TRADUTORES, value=TRADUTORES[0], label="Tradutor")
            with gr.Row(equal_height=True):
                pasta_destino = gr.Textbox(value=pasta_padrao, scale=4,
                                           label="Pasta onde salvar os vídeos")
                botao_pasta = gr.Button("📁 Escolher", scale=1)
            with gr.Row():
                botao = gr.Button("Dublar", variant="primary")
                botao_parar = gr.Button("Parar", variant="stop")
                botao_limpar = gr.Button("🧹 Limpar")
        with gr.Column():
            video_saida = gr.Video(label="Vídeo dublado")
            arquivos_saida = gr.File(label="Arquivos para baixar", file_count="multiple",
                                     interactive=False)
            tempo_saida = gr.Markdown()
    log_saida = gr.Textbox(label="Log do processo", lines=12, max_lines=12,
                           autoscroll=True, interactive=False)
    relogio_log = gr.Timer(1, active=False)

    modo.change(trocar_modo, modo, [opcoes_voz, botao])
    botao_pasta.click(escolher_pasta, pasta_destino, pasta_destino)
    botao_ouvir.click(lambda nome: gr.Audio(ouvir_voz(nome), visible=True),
                      voz, amostra)
    relogio_log.tick(mostrar_log, None, log_saida, show_progress="hidden")
    botao.click(lambda: gr.Timer(active=True), None, relogio_log).then(
        dublar, [url, arquivo, modo, voz, modelo, tradutor, pasta_destino],
        [video_saida, arquivos_saida, tempo_saida],
        show_progress_on=[video_saida]).then(
        fim_do_log, None, [log_saida, relogio_log])
    botao_parar.click(parar, queue=False)
    botao_limpar.click(limpar, None, [url, arquivo, video_saida, arquivos_saida,
                                      tempo_saida, log_saida, amostra], queue=False)

def ja_aberto(endereco):
    try:
        urllib.request.urlopen(endereco, timeout=2)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    endereco = "http://127.0.0.1:7860"
    if ja_aberto(endereco):  # clicou no atalho de novo: só mostra o app que já roda
        print(f"O dublador já está aberto em {endereco}")
        webbrowser.open(endereco)
    else:
        app.launch(server_port=7860, inbrowser=True,
                   favicon_path=str(Path(__file__).parent / "icone.png"))
