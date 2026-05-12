import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import torch
import whisper

FORMATO_SAIDA = ".mp3"
# Extensoes comuns para filtro rapido; a deteccao final e feita por ffprobe.
EXTENSOES_AUDIO_COMUNS = {
    ".mp3",
    ".opus",
    ".ogg",
    ".mp4",
    ".m4a",
    ".wav",
    ".flac",
    ".webm",
    ".wma",
    ".aac",
    ".amr",
    ".3gp",
    ".mkv",
    ".mov",
}
MSG_FFMPEG = (
    "FFmpeg não encontrado. O Whisper e a conversão de áudio dependem do FFmpeg.\n"
    "Instale e adicione ao PATH: https://ffmpeg.org/download.html\n"
    "No Windows: baixe em https://www.gyan.dev/ffmpeg/builds/ e extraia a pasta 'bin' no PATH."
)
MSG_YT_DLP = "Para transcrever links do YouTube, instale: pip install yt-dlp"


def _linha_e_url_youtube(linha: str) -> bool:
    s = linha.strip()
    if not s or s.startswith("#"):
        return False
    if not s.lower().startswith("http"):
        return False
    lower = s.lower()
    return "youtube.com/" in lower or "youtu.be/" in lower


def ler_urls_youtube_txt(caminho_txt: Path) -> list[str]:
    """Lê um .txt: uma URL do YouTube por linha; linhas vazias e # comentários são ignoradas."""
    if not caminho_txt.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_txt}")
    conteudo = caminho_txt.read_text(encoding="utf-8", errors="replace")
    urls: list[str] = []
    for raw in conteudo.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _linha_e_url_youtube(line):
            primeira = line.split()[0]
            urls.append(primeira)
        else:
            trecho = line if len(line) <= 120 else line[:117] + "..."
            print(f"Aviso: linha ignorada (não parece URL do YouTube): {trecho}")
    return urls


def sanitizar_titulo_para_nome_ficheiro(titulo: str | None, fallback: str) -> str:
    """Prepara o título do vídeo para usar como nome de ficheiro no Windows."""
    t = (titulo or "").replace("\r", " ").replace("\n", " ").strip()
    if not t:
        t = fallback
    # Caracteres inválidos em nomes de ficheiro no Windows
    t = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", t)
    t = re.sub(r"\s+", " ", t).strip(" .")
    if len(t) > 180:
        t = t[:180].rstrip(" .")
    if not t:
        t = fallback
    return t


def nome_base_txt_youtube(pasta: Path, titulo: str | None, video_id: str) -> str:
    """
    Nome do .txt sem extensão: título sanitizado; se já existir ficheiro, acrescenta [id].
    """
    vid = re.sub(r'[<>:"/\\|?*]', "_", str(video_id or "video"))
    stem = sanitizar_titulo_para_nome_ficheiro(titulo, vid)
    if not (pasta / f"{stem}.txt").exists():
        return stem
    stem2 = f"{stem} [{vid}]"
    if len(stem2) > 200:
        stem2 = stem2[:200].rstrip(" .")
    n = 2
    while (pasta / f"{stem2}.txt").exists():
        stem2 = f"{stem} [{vid}] ({n})"
        n += 1
    return stem2


def baixar_audio_youtube_mp3(url: str, pasta_destino: Path) -> tuple[Path, dict]:
    """
    Baixa o melhor áudio do vídeo, extrai para MP3 com FFmpeg.
    Devolve (caminho_mp3, metadados) — o ficheiro continua nomeado pelo ID do vídeo
    (estável para o yt-dlp); use o título em `metadados` para o nome da transcrição.
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise ImportError(MSG_YT_DLP) from e

    pasta_destino = pasta_destino.resolve()
    pasta_destino.mkdir(parents=True, exist_ok=True)
    base = str(pasta_destino / "%(id)s")

    opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": base + ".%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        raise RuntimeError(f"Falha ao baixar áudio do YouTube: {e}") from e

    video_id = (info or {}).get("id") or "audio"
    # Remove caracteres inválidos no Windows (raro em IDs do YouTube).
    video_id = re.sub(r'[<>:"/\\|?*]', "_", str(video_id))
    caminho_mp3 = pasta_destino / f"{video_id}.mp3"
    if not caminho_mp3.is_file():
        raise RuntimeError(
            f"O download terminou mas o MP3 não foi encontrado em: {caminho_mp3}\n"
            "Confira se o FFmpeg está no PATH (o yt-dlp usa o FFmpeg para extrair o áudio)."
        )
    return caminho_mp3, info if isinstance(info, dict) else {}


def transcrever_txt_youtube_links(
    caminho_txt: Path,
    saida_unica: str | None = None,
) -> list[tuple[str, str]]:
    """
    Para cada URL em `caminho_txt` (mesma pasta = pasta de saída): baixa MP3, transcreve,
    apaga o MP3 e grava `{título do vídeo}.txt` na mesma pasta (título sanitizado).
    Se `saida_unica` for informado, não grava .txt por vídeo; junta tudo nesse arquivo ao final.
    """
    urls = ler_urls_youtube_txt(caminho_txt)
    if not urls:
        print(f"Nenhuma URL do YouTube encontrada em: {caminho_txt}")
        return []

    pasta = caminho_txt.parent.resolve()
    print(f"Pasta de trabalho (áudio temporário e .txt): {pasta}")
    print(f"Encontrada(s) {len(urls)} URL(s) do YouTube.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")
    print("Carregando modelo Whisper...")
    model = whisper.load_model("turbo", device=device)

    resultados: list[tuple[str, str]] = []
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")
        mp3_path: Path | None = None
        try:
            print("Baixando áudio (MP3)...")
            mp3_path, meta = baixar_audio_youtube_mp3(url, pasta)
            print(f"Transcrevendo: {mp3_path.name}...")
            texto = _transcrever_para_texto(str(mp3_path), model)
            resultados.append((url, texto))
            if saida_unica is None:
                titulo = meta.get("title") or meta.get("fulltitle")
                id_video = str(meta.get("id") or mp3_path.stem)
                stem_txt = nome_base_txt_youtube(pasta, titulo if isinstance(titulo, str) else None, id_video)
                saida_item = pasta / f"{stem_txt}.txt"
                with open(saida_item, "w", encoding="utf-8") as f:
                    f.write(texto)
                print(f"Transcrição salva em: {saida_item}")
        except (OSError, ValueError, ImportError, RuntimeError) as e:
            print(f"Erro: {e}")
        finally:
            if mp3_path is not None and mp3_path.is_file():
                mp3_path.unlink()
                print(f"MP3 temporário removido: {mp3_path.name}")

    if saida_unica and resultados:
        path_saida = Path(saida_unica)
        if path_saida.suffix.lower() != ".txt":
            path_saida = path_saida.with_suffix(".txt")
        path_saida.parent.mkdir(parents=True, exist_ok=True)
        with open(path_saida, "w", encoding="utf-8") as f:
            for url_item, texto in resultados:
                f.write(f"--- {url_item} ---\n\n")
                f.write(texto)
                f.write("\n\n")
        print(f"\nTranscrição única salva em: {path_saida}")
    return resultados


def transcrever_link_youtube_cli(
    url: str,
    pasta_trabalho: Path,
    nome_saida: str | None = None,
) -> str:
    """
    Uma única URL do YouTube: baixa MP3 em `pasta_trabalho`, transcreve, apaga o MP3 e grava o .txt.
    Com `nome_saida` None, o ficheiro usa o título do vídeo (sanitizado) nessa pasta.
    """
    pasta_trabalho = pasta_trabalho.resolve()
    pasta_trabalho.mkdir(parents=True, exist_ok=True)
    print(f"Pasta de trabalho: {pasta_trabalho}")
    print(url)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")
    print("Carregando modelo Whisper...")
    model = whisper.load_model("turbo", device=device)

    mp3_path: Path | None = None
    try:
        print("Baixando áudio (MP3)...")
        mp3_path, meta = baixar_audio_youtube_mp3(url, pasta_trabalho)
        print(f"Transcrevendo: {mp3_path.name}...")
        texto = _transcrever_para_texto(str(mp3_path), model)
        if nome_saida is None:
            titulo = meta.get("title") or meta.get("fulltitle")
            id_video = str(meta.get("id") or mp3_path.stem)
            stem_txt = nome_base_txt_youtube(
                pasta_trabalho,
                titulo if isinstance(titulo, str) else None,
                id_video,
            )
            saida_path = pasta_trabalho / f"{stem_txt}.txt"
        else:
            p = Path(nome_saida).expanduser()
            if p.is_absolute():
                saida_path = p if p.suffix.lower() == ".txt" else p.with_suffix(".txt")
            else:
                nome = nome_saida if nome_saida.endswith(".txt") else f"{Path(nome_saida).stem}.txt"
                saida_path = pasta_trabalho / nome
        saida_path.parent.mkdir(parents=True, exist_ok=True)
        with open(saida_path, "w", encoding="utf-8") as f:
            f.write(texto)
        print(f"Transcrição salva em: {saida_path}")
        return texto
    finally:
        if mp3_path is not None and mp3_path.is_file():
            mp3_path.unlink()
            print(f"MP3 temporário removido: {mp3_path.name}")


def ffmpeg_disponivel() -> bool:
    """Verifica se o FFmpeg está disponível no sistema."""
    return shutil.which("ffmpeg") is not None


def ffprobe_disponivel() -> bool:
    """Verifica se o ffprobe está disponível no sistema."""
    return shutil.which("ffprobe") is not None


def arquivo_tem_stream_audio(caminho_arquivo: Path) -> bool:
    """
    Detecta se o arquivo possui pelo menos um stream de audio.
    """
    if not caminho_arquivo.is_file():
        return False
    if not ffprobe_disponivel():
        return caminho_arquivo.suffix.lower() in EXTENSOES_AUDIO_COMUNS
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(caminho_arquivo),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except OSError:
        return caminho_arquivo.suffix.lower() in EXTENSOES_AUDIO_COMUNS
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0 and "audio" in proc.stdout.lower()


def converter_para_mp3(caminho_arquivo: str) -> tuple[str, bool]:
    """
    Converte qualquer arquivo com stream de audio para MP3.
    Retorna (caminho_mp3, foi_convertido).
    """
    caminho = Path(caminho_arquivo)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    if not arquivo_tem_stream_audio(caminho):
        raise ValueError(f"O arquivo não possui stream de áudio: {caminho.name}")

    extensao = caminho.suffix.lower()
    if extensao == ".mp3":
        return str(caminho.resolve()), False

    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "Para conversão automática, instale: pip install pydub\n"
            "E tenha o FFmpeg instalado no sistema (https://ffmpeg.org)"
        )

    with tempfile.NamedTemporaryFile(
        prefix=f"{caminho.stem}_",
        suffix=FORMATO_SAIDA,
        delete=False,
        dir=str(caminho.parent),
    ) as tmp:
        arquivo_mp3 = Path(tmp.name)

    print(f"Convertendo {caminho.name} para MP3...")
    try:
        audio = AudioSegment.from_file(str(caminho))
    except Exception as e:
        raise ValueError(
            f"Não foi possível decodificar '{caminho.name}'. "
            "Verifique se o arquivo é áudio válido e se o FFmpeg está no PATH."
        ) from e
    audio.export(str(arquivo_mp3), format="mp3", bitrate="192k")
    print(f"Salvo em: {arquivo_mp3.name}")
    return str(arquivo_mp3.resolve()), True


def extrair_zip(origem: Path, destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(origem, "r") as zf:
        zf.extractall(destino)


def extrair_rar(origem: Path, destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    seven = shutil.which("7z") or shutil.which("7za")
    if seven:
        subprocess.run(
            [seven, "x", str(origem), f"-o{destino}{os.sep}", "-y"],
            check=True,
        )
        return
    unrar = shutil.which("UnRAR") or shutil.which("unrar")
    if unrar:
        subprocess.run([unrar, "x", "-o+", str(origem), str(destino) + os.sep], check=True)
        return
    try:
        import rarfile  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "Para .rar: instale o 7-Zip (7z no PATH), UnRAR, ou pip install rarfile + ferramenta UnRAR."
        ) from e
    with rarfile.RarFile(origem) as rf:
        rf.extractall(destino)


def preparar_caminho_entrada(caminho: Path) -> Path:
    """
    Se for .zip ou .rar, extrai para uma pasta com o mesmo nome (sem extensão),
    ao lado do arquivo, e devolve esse diretório. Caso contrário devolve o próprio caminho.
    """
    if caminho.is_dir():
        return caminho.resolve()
    if not caminho.is_file():
        raise FileNotFoundError(f"Não encontrado: {caminho}")
    caminho = caminho.resolve()
    suf = caminho.suffix.lower()
    if suf == ".zip":
        destino = caminho.parent / caminho.stem
        destino.mkdir(parents=True, exist_ok=True)
        print(f"Extraindo ZIP para: {destino}")
        extrair_zip(caminho, destino)
        return destino
    if suf == ".rar":
        destino = caminho.parent / caminho.stem
        destino.mkdir(parents=True, exist_ok=True)
        print(f"Extraindo RAR para: {destino}")
        extrair_rar(caminho, destino)
        return destino
    return caminho


def listar_audios_pasta(pasta: str) -> list[Path]:
    """Retorna arquivos de áudio na pasta (recursivo), em ordem alfabética pelo caminho."""
    caminho = Path(pasta)
    if not caminho.is_dir():
        raise NotADirectoryError(f"Não é uma pasta: {pasta}")
    arquivos = [f for f in caminho.rglob("*") if f.is_file() and arquivo_tem_stream_audio(f)]
    return sorted(arquivos, key=lambda f: str(f).lower())


def _transcrever_para_texto(arquivo_entrada: str, model: whisper.Whisper) -> str:
    """Converte para MP3 se necessário, transcreve com o modelo e retorna apenas o texto."""
    arquivo_mp3, foi_convertido = converter_para_mp3(arquivo_entrada)
    result = model.transcribe(arquivo_mp3)

    if foi_convertido:
        caminho_mp3 = Path(arquivo_mp3)
        if caminho_mp3.exists():
            caminho_mp3.unlink()
            print(f"MP3 removido após transcrição: {caminho_mp3.name}")

    return result["text"].strip()


def transcrever_e_salvar(
    arquivo_entrada: str,
    saida_txt: str | None = None,
    model: whisper.Whisper | None = None,
) -> str:
    """
    Transcreve áudio/vídeo e salva o texto em um arquivo .txt.
    Se model for passado, reutiliza o modelo (útil para transcrever vários arquivos).
    Retorna o texto transcrito.
    """
    caminho_entrada = Path(arquivo_entrada)

    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Dispositivo: {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")
        print("Carregando modelo Whisper...")
        model = whisper.load_model("turbo", device=device)

    print(f"Transcrevendo: {caminho_entrada.name}...")
    texto = _transcrever_para_texto(arquivo_entrada, model)

    if saida_txt is None:
        saida_txt = str(caminho_entrada.with_suffix(".txt"))
    if not saida_txt.endswith(".txt"):
        saida_txt = f"{Path(saida_txt).stem}.txt"

    with open(saida_txt, "w", encoding="utf-8") as f:
        f.write(texto)

    print(f"Transcrição salva em: {saida_txt}")
    return texto


def transcrever_pasta(
    pasta: str,
    saida_unica: str | None = None,
) -> list[tuple[str, str]]:
    """
    Transcreve todos os arquivos de áudio da pasta (ordem alfabética por nome).
    Se saida_unica for informado, grava todas as transcrições em um único .txt, em ordem.
    Caso contrário, salva cada uma em .txt no mesmo diretório.
    Retorna lista de (arquivo_entrada, texto_transcrito).
    """
    arquivos = listar_audios_pasta(pasta)
    if not arquivos:
        print(f"Nenhum arquivo de áudio encontrado em: {pasta}")
        return []

    print(f"Encontrados {len(arquivos)} arquivo(s) de áudio (ordem alfabética).")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {'GPU (CUDA)' if device == 'cuda' else 'CPU'}")
    print("Carregando modelo Whisper...")
    model = whisper.load_model("turbo", device=device)

    resultados = []
    for i, caminho in enumerate(arquivos, 1):
        print(f"\n[{i}/{len(arquivos)}] ", end="")
        try:
            texto = _transcrever_para_texto(str(caminho), model)
            resultados.append((str(caminho), texto))
            if saida_unica is None:
                saida_txt = str(Path(caminho).with_suffix(".txt"))
                with open(saida_txt, "w", encoding="utf-8") as f:
                    f.write(texto)
                print(f"Transcrição salva em: {saida_txt}")
        except (OSError, ValueError, ImportError) as e:
            print(f"Erro em {caminho.name}: {e}")

    if saida_unica and resultados:
        path_saida = Path(saida_unica)
        if not path_saida.suffix.lower() == ".txt":
            path_saida = path_saida.with_suffix(".txt")
        with open(path_saida, "w", encoding="utf-8") as f:
            for caminho, texto in resultados:
                nome = Path(caminho).name
                f.write(f"--- {nome} ---\n\n")
                f.write(texto)
                f.write("\n\n")
        print(f"\nTranscrição única salva em: {path_saida}")
    return resultados


def _formatar_duracao(segundos: float) -> str:
    """Formata duração em segundos para exibição (ex: '2 min 35 s' ou '45.2 s')."""
    if segundos < 60:
        return f"{segundos:.1f} s"
    mins = int(segundos // 60)
    secs = segundos % 60
    if secs < 0.05:
        return f"{mins} min"
    return f"{mins} min {secs:.1f} s"


if __name__ == "__main__":
    import sys

    argv_rest = [a for a in sys.argv[1:] if a not in ("--unico", "-u")]
    unico = len(argv_rest) < len(sys.argv[1:])

    entrada_raw = argv_rest[0] if argv_rest else "audio.mp3"
    segundo_arg = argv_rest[1] if len(argv_rest) > 1 else None

    nome_saida: str | None = None
    saida_unica: str | None = None

    if not ffmpeg_disponivel():
        print("Aviso: FFmpeg não está no PATH. A transcrição pode falhar.")
        print(MSG_FFMPEG)

    inicio = time.perf_counter()
    try:
        entrada_norm = entrada_raw.strip()
        token_yt = entrada_norm.split()[0] if entrada_norm else ""
        if _linha_e_url_youtube(token_yt):
            if unico:
                print(
                    "Aviso: --unico com um único link do YouTube não altera o resultado; ignorado."
                )
            texto = transcrever_link_youtube_cli(
                token_yt,
                Path.cwd(),
                nome_saida=segundo_arg,
            )
            print("\n--- Transcrição ---")
            print(texto)
        else:
            caminho = preparar_caminho_entrada(Path(entrada_raw).expanduser())

            if caminho.is_dir():
                if unico:
                    if segundo_arg:
                        p = Path(segundo_arg)
                        if p.is_absolute():
                            out = p if p.suffix.lower() == ".txt" else p.with_suffix(".txt")
                            saida_unica = str(out)
                        else:
                            nome_txt = (
                                segundo_arg
                                if segundo_arg.endswith(".txt")
                                else f"{Path(segundo_arg).stem}.txt"
                            )
                            saida_unica = str(caminho / nome_txt)
                    else:
                        saida_unica = str(caminho / "transcricao_completa.txt")
                resultados = transcrever_pasta(str(caminho), saida_unica=saida_unica)
                print(f"\n--- Concluído: {len(resultados)} arquivo(s) transcrito(s) ---")
            elif caminho.is_file() and caminho.suffix.lower() == ".txt":
                urls_yt = ler_urls_youtube_txt(caminho)
                if urls_yt:
                    if unico:
                        if segundo_arg:
                            p = Path(segundo_arg)
                            if p.is_absolute():
                                out = p if p.suffix.lower() == ".txt" else p.with_suffix(".txt")
                                saida_unica = str(out)
                            else:
                                nome_txt = (
                                    segundo_arg
                                    if segundo_arg.endswith(".txt")
                                    else f"{Path(segundo_arg).stem}.txt"
                                )
                                saida_unica = str(caminho.parent / nome_txt)
                        else:
                            saida_unica = str(caminho.parent / "transcricao_completa.txt")
                    elif segundo_arg:
                        print(
                            "Aviso: com lista de links do YouTube, o segundo argumento só é usado com --unico "
                            "(nome do .txt de saída única)."
                        )
                    resultados = transcrever_txt_youtube_links(caminho, saida_unica=saida_unica)
                    print(f"\n--- Concluído: {len(resultados)} vídeo(s) transcrito(s) ---")
                else:
                    nome_saida = segundo_arg
                    if unico:
                        print("Aviso: --unico aplica-se a pastas ou a .txt com URLs do YouTube; ignorado.")
                    texto = transcrever_e_salvar(str(caminho), nome_saida)
                    print("\n--- Transcrição ---")
                    print(texto)
            else:
                nome_saida = segundo_arg
                if unico:
                    print("Aviso: --unico aplica-se a pastas ou a .txt com URLs do YouTube; ignorado.")
                texto = transcrever_e_salvar(str(caminho), nome_saida)
                print("\n--- Transcrição ---")
                print(texto)
    except OSError as e:
        msg = str(e).lower()
        if (getattr(e, "winerror", None) == 2) or "não pode encontrar" in msg or "cannot find" in msg:
            print("Erro: arquivo não encontrado (provavelmente FFmpeg).")
            print(MSG_FFMPEG)
        else:
            print(f"Erro: {e}")
        sys.exit(1)
    except (ValueError, ImportError, NotADirectoryError, RuntimeError) as e:
        print(f"Erro: {e}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao extrair arquivo compactado (código {e.returncode}).")
        if e.stderr:
            print(e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else e.stderr)
        sys.exit(1)
    finally:
        duracao = time.perf_counter() - inicio
        print(f"\nTempo total: {_formatar_duracao(duracao)}")
