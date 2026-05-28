"""
buscar.py — Busca desenhos técnicos similares a um input (PDF ou imagem JPG/PNG).

Uso:
    python buscar.py --input meu_projeto.pdf --indice ./indice --top 10
    python buscar.py --input referencia.jpg  --indice ./indice --top 5
    python buscar.py --texto "planta baixa residencial" --indice ./indice --top 10

Dependências:
    pip install torch torchvision open_clip_torch pymupdf pillow numpy
"""

import argparse
import json
from pathlib import Path

import fitz
import numpy as np
import torch
import open_clip
from PIL import Image


MODELO = "ViT-L-14"
PRETRAINED = "openai"


def carregar_modelo(device):
    model, _, preprocess = open_clip.create_model_and_transforms(MODELO, pretrained=PRETRAINED)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODELO)
    return model, preprocess, tokenizer


def embedding_de_imagem(pil_img, model, preprocess, device):
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


def embedding_de_texto(texto, model, tokenizer, device):
    tokens = tokenizer([texto]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


def carregar_imagem_input(caminho: str, dpi=150):
    """Carrega a imagem de busca — aceita JPG, PNG ou PDF (usa primeira página)."""
    path = Path(caminho)
    sufixo = path.suffix.lower()

    if sufixo == ".pdf":
        doc = fitz.open(str(path))
        page = doc[0]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    elif sufixo in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"):
        return Image.open(str(path)).convert("RGB")
    else:
        raise ValueError(f"Formato não suportado: {sufixo}. Use PDF, JPG ou PNG.")


def buscar(indice_dir: str, input_path: str = None, texto: str = None, top_k: int = 10):
    indice_dir = Path(indice_dir)
    embeddings_path = indice_dir / "embeddings.npy"
    metadados_path = indice_dir / "metadados.json"

    if not embeddings_path.exists() or not metadados_path.exists():
        print(f"[ERRO] Índice não encontrado em '{indice_dir}'.")
        print("       Execute primeiro: python indexar.py --pasta <sua_pasta> --saida ./indice")
        return

    print("[INFO] Carregando índice...")
    matriz = np.load(str(embeddings_path))  # (N, D)
    with open(metadados_path, "r", encoding="utf-8") as f:
        metadados = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Carregando modelo CLIP (dispositivo: {device})...")
    model, preprocess, tokenizer = carregar_modelo(device)

    # Gera embedding do input
    if input_path:
        print(f"[INFO] Processando input: {input_path}")
        img = carregar_imagem_input(input_path)
        query_emb = embedding_de_imagem(img, model, preprocess, device)
        modo = f"imagem '{Path(input_path).name}'"
    elif texto:
        print(f"[INFO] Buscando por texto: '{texto}'")
        query_emb = embedding_de_texto(texto, model, tokenizer, device)
        modo = f"texto '{texto}'"
    else:
        print("[ERRO] Forneça --input ou --texto.")
        return

    # Similaridade de cosseno (dot product, pois já normalizamos)
    scores = (matriz @ query_emb.T).flatten()  # (N,)

    # Ordena por score descendente
    ranking = np.argsort(scores)[::-1]

    # Remove duplicatas de arquivo (mantém a página com maior score por arquivo)
    vistos = {}
    for idx in ranking:
        arquivo = metadados[idx]["arquivo"]
        if arquivo not in vistos:
            vistos[arquivo] = (idx, float(scores[idx]))
        if len(vistos) >= top_k:
            break

    # Exibe resultados
    print(f"\n{'─'*60}")
    print(f"  Top {top_k} projetos similares a {modo}")
    print(f"{'─'*60}")
    for rank, (arquivo, (idx, score)) in enumerate(vistos.items(), start=1):
        meta = metadados[idx]
        barra = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        print(f"\n  #{rank:>2}  Score: {score:.4f}  {barra}")
        print(f"       Arquivo : {arquivo}")
        print(f"       Página  : {meta['pagina'] + 1}")
        print(f"       Caminho : {meta['caminho_completo']}")

    print(f"\n{'─'*60}")
    print(f"  {len(vistos)} resultado(s) exibido(s) de {len(metadados)} páginas indexadas.")
    print(f"{'─'*60}\n")

    # Retorna lista para uso programático
    return [
        {
            "rank": i + 1,
            "score": score,
            "arquivo": arq,
            "pagina": metadados[idx]["pagina"] + 1,
            "caminho_completo": metadados[idx]["caminho_completo"],
        }
        for i, (arq, (idx, score)) in enumerate(vistos.items())
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Busca desenhos técnicos similares com CLIP.")
    parser.add_argument("--indice", default="./indice", help="Pasta do índice (default: ./indice)")

    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--input", help="Arquivo de busca: PDF ou imagem (JPG/PNG)")
    grupo.add_argument("--texto", help="Busca por descrição textual (ex: 'planta baixa')")

    parser.add_argument("--top", type=int, default=10, help="Quantos resultados exibir (default: 10)")
    args = parser.parse_args()

    buscar(args.indice, input_path=args.input, texto=args.texto, top_k=args.top)
