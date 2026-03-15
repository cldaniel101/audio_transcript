# whisper_test

Script simples para transcrever áudio localmente usando [OpenAI Whisper](https://github.com/openai/whisper).

Criei este projeto para uso pessoal, com foco em praticidade: apontar um arquivo ou uma pasta, deixar o Whisper rodar localmente e salvar o resultado em `.txt`. Mesmo assim, ele está organizado o suficiente para ser útil para outras pessoas que queiram algo direto, sem interface gráfica e sem muita configuração.

## O que este projeto faz

- Transcreve um único arquivo de áudio para texto.
- Transcreve todos os arquivos de áudio de uma pasta.
- Converte automaticamente arquivos suportados para `.mp3` antes da transcrição, quando necessário.
- Salva uma transcrição por arquivo ou junta tudo em um único `.txt`.
- Usa GPU automaticamente quando o PyTorch com CUDA estiver disponível.

## Formatos suportados

Entrada:

- `.mp3`
- `.ogg`
- `.mp4`
- `.m4a`
- `.wav`
- `.flac`
- `.webm`
- `.wma`
- `.aac`

Saída:

- Transcrição em arquivo `.txt`
- Conversão intermediária para `.mp3` quando o arquivo original não estiver em MP3

## Como funciona

O script principal está em `main.py`. Ele:

1. Verifica se o arquivo informado precisa ser convertido para MP3.
2. Usa o modelo `turbo` do Whisper para fazer a transcrição.
3. Detecta automaticamente se deve usar `CPU` ou `GPU (CUDA)`.
4. Salva o texto transcrito em um arquivo `.txt`.

Se a entrada for uma pasta, o script procura todos os arquivos de áudio suportados, ordena em ordem alfabética e transcreve cada um deles.

## Requisitos

- Python 3.10+ recomendado
- [FFmpeg](https://ffmpeg.org/download.html) instalado e disponível no `PATH`
- Dependências do `requirements.txt`

## Instalação

### 1. Clonar o repositório

```powershell
git clone <URL_DO_REPOSITORIO>
cd whisper_test
```

### 2. Criar e ativar ambiente virtual

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Instalar dependências

O arquivo `requirements.txt` reflete o ambiente que usei localmente e inclui uma build específica do PyTorch com CUDA. Em algumas máquinas, especialmente sem GPU NVIDIA compatível, pode ser melhor instalar o PyTorch separadamente e depois instalar o restante.

Se quiser tentar exatamente como está no repositório:

```powershell
pip install -r requirements.txt
```

Se o PyTorch falhar na instalação, a abordagem mais segura é:

1. instalar o PyTorch conforme seu ambiente em https://pytorch.org/get-started/locally/
2. instalar o restante das dependências manualmente

Exemplo mínimo:

```powershell
pip install openai-whisper pydub
```

## Instalar o FFmpeg

O Whisper depende do FFmpeg para ler e processar áudio. A conversão automática com `pydub` também depende dele.

Se estiver no Windows, uma opção prática é baixar um build em:

- https://www.gyan.dev/ffmpeg/builds/

Depois, adicione a pasta `bin` do FFmpeg ao `PATH`.

Site oficial:

- https://ffmpeg.org/download.html

## Uso

### Transcrever um arquivo

Se nenhum argumento for informado, o script tenta usar `audio.mp3` no diretório atual:

```powershell
python main.py
```

Para informar um arquivo específico:

```powershell
python main.py "gravacao.m4a"
```

Para definir o nome do arquivo de saída:

```powershell
python main.py "gravacao.m4a" "resultado.txt"
```

Comportamento:

- Se o arquivo não for `.mp3`, ele pode ser convertido antes da transcrição.
- O texto também é exibido no terminal ao final.
- O arquivo `.txt` é salvo no disco.

### Transcrever todos os áudios de uma pasta

```powershell
python main.py ".\audios"
```

Nesse modo, o script:

- encontra os arquivos suportados da pasta
- carrega o modelo uma vez só
- gera um `.txt` para cada arquivo

### Gerar um único arquivo com todas as transcrições

```powershell
python main.py ".\audios" --unico
```

Ou definindo o nome do arquivo final:

```powershell
python main.py ".\audios" --unico "transcricao_completa.txt"
```

Nesse caso, o resultado consolidado é salvo em um único `.txt`, com separação por nome de arquivo.

## Exemplo de estrutura de saída

Entrada:

```text
audios/
  aula1.m4a
  aula2.ogg
```

Saída possível:

```text
audios/
  aula1.m4a
  aula1.mp3
  aula1.txt
  aula2.ogg
  aula2.mp3
  aula2.txt
```

Ou, usando `--unico`:

```text
audios/
  aula1.m4a
  aula1.mp3
  aula2.ogg
  aula2.mp3
  transcricao_completa.txt
```

## GPU / CUDA

Se o PyTorch estiver instalado com suporte a CUDA, o script usa GPU automaticamente.

Se aparecer `Dispositivo: CPU`, então sua instalação atual do PyTorch provavelmente não tem suporte a CUDA.

Há instruções mais detalhadas em `GPU.md`.

## Limitações e observações

- Este projeto é focado em transcrição simples via terminal.
- Não há interface gráfica.
- Não há diarização de falantes.
- Não há tradução automática no fluxo atual.
- A qualidade da transcrição depende bastante da clareza do áudio.
- Arquivos convertidos para `.mp3` são mantidos no diretório; o script não remove esses arquivos depois.

## Quando esse projeto é útil

- Transcrever áudios pessoais rapidamente
- Converter gravações de aula, reunião ou nota de voz em texto
- Processar uma pasta inteira sem precisar abrir programa com interface
- Rodar tudo localmente, sem enviar arquivo para serviço online

## Possíveis melhorias futuras

- adicionar argumentos de linha de comando mais claros com `argparse`
- permitir escolher o modelo do Whisper via terminal
- exportar também em formatos como `.srt`
- permitir idioma explícito como parâmetro
- evitar reconversões desnecessárias quando o `.mp3` já existir

## Licença

Ainda não defini uma licença para este repositório. Se você pretende reutilizar o código publicamente, vale adicionar uma licença explícita.
