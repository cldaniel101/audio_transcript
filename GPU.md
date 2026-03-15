# Usar Whisper com GPU (NVIDIA CUDA)

O script usa GPU automaticamente quando o PyTorch está instalado com suporte a CUDA. Se aparecer **"Dispositivo: CPU"**, o PyTorch atual é só para CPU.

## 1. Verificar sua GPU e driver

No PowerShell:

```powershell
nvidia-smi
```

Anote a versão do **Driver** e da **CUDA** (se aparecer). Você precisa de um driver NVIDIA atualizado.

## 2. Reinstalar PyTorch com CUDA

Ative o ambiente virtual e instale o PyTorch com CUDA (escolha **uma** linha conforme sua versão de CUDA):

**CUDA 12.4** (recomendado para a maioria):

```powershell
.\.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**CUDA 12.1**:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8** (placas/drivers mais antigos):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

O site oficial com as opções atuais: https://pytorch.org/get-started/locally/

## 3. Confirmar que a GPU está disponível

No mesmo ambiente:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Se aparecer `CUDA: True` e o nome da placa, pode rodar o script; a saída deve mostrar **"Dispositivo: GPU (CUDA)"**.

## 4. Rodar o script

```powershell
python main.py
```

O aviso *"FP16 is not supported on CPU"* deixa de aparecer quando estiver usando GPU.
