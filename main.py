import os
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
        else:
            nome_saida = segundo_arg
            if unico:
                print("Aviso: --unico aplica-se apenas ao transcrever uma pasta; ignorado.")
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
