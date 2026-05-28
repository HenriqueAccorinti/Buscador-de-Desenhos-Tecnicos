"""
indexar.py — Converte PDFs em imagens e gera embeddings CLIP para busca por similaridade.

Uso:
    python indexar.py --pasta ./meus_projetos --saida ./indice

Dependências:
    pip install torch torchvision open_clip_torch pymupdf pillow numpy tqdm
"""

import argparse
import json
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────
# Configurações
# ─────────────────────────────────────────────
DPI = 150          # Resolução da conversão PDF→imagem (150 é bom para desenhos técnicos)
MAX_PAGINAS = 1    # Quantas páginas por PDF indexar (1 = só a primeira; use None para todas)
BATCH_SIZE = 16    # Quantos embeddings calcular por vez (reduza se der erro de memória)
MODELO = "ViT-L-14"
PRETRAINED = "openai"


def carregar_modelo(device):
    print(f"[CLIP] Carregando modelo {MODELO} ({PRETRAINED})...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODELO, pretrained=PRETRAINED)
    model = model.to(device).eval()
    print("[CLIP] Modelo carregado.\n")
    return model, preprocess


def pdf_para_imagens(caminho_pdf: Path, max_paginas=MAX_PAGINAS, dpi=DPI):
    """Converte páginas de um PDF em objetos PIL.Image."""
    imagens = []
    try:
        doc = fitz.open(str(caminho_pdf))
        n = len(doc) if max_paginas is None else min(max_paginas, len(doc))
        for i in range(n):
            page = doc[i]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            imagens.append((i, img))
        doc.close()
    except Exception as e:
        print(f"  [ERRO] Falha ao processar {caminho_pdf.name}: {e}")
    return imagens


def calcular_embeddings(imagens_pil, model, preprocess, device):
    """Calcula embeddings CLIP para uma lista de PIL.Images."""
    tensors = torch.stack([preprocess(img) for img in imagens_pil]).to(device)
    with torch.no_grad():
        features = model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)  # normaliza
    return features.cpu().numpy().astype(np.float32)


def indexar(pasta_pdfs: str, pasta_saida: str):
    pasta_pdfs = Path(pasta_pdfs)
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Usando dispositivo: {device}")
    model, preprocess = carregar_modelo(device)

    # Coleta todos os PDFs
    pdfs = sorted(pasta_pdfs.rglob("*.pdf")) + sorted(pasta_pdfs.rglob("*.PDF"))
    if not pdfs:
        print(f"[AVISO] Nenhum PDF encontrado em: {pasta_pdfs}")
        sys.exit(1)
    print(f"[INFO] {len(pdfs)} PDFs encontrados.\n")

    todos_embeddings = []
    todos_metadados = []

    for pdf_path in tqdm(pdfs, desc="Indexando PDFs"):
        paginas = pdf_para_imagens(pdf_path)
        if not paginas:
            continue

        # Processa em batches
        for inicio in range(0, len(paginas), BATCH_SIZE):
            lote = paginas[inicio:inicio + BATCH_SIZE]
            indices_pag = [p[0] for p in lote]
            imagens = [p[1] for p in lote]

            embeddings = calcular_embeddings(imagens, model, preprocess, device)

            for emb, pag_idx in zip(embeddings, indices_pag):
                todos_embeddings.append(emb)
                todos_metadados.append({
                    "arquivo": str(pdf_path.relative_to(pasta_pdfs)),
                    "caminho_completo": str(pdf_path),
                    "pagina": pag_idx,
                })

    if not todos_embeddings:
        print("[ERRO] Nenhum embedding gerado. Verifique os PDFs.")
        sys.exit(1)

    # Salva índice
    matriz = np.stack(todos_embeddings)  # shape: (N, embedding_dim)
    np.save(pasta_saida / "embeddings.npy", matriz)

    with open(pasta_saida / "metadados.json", "w", encoding="utf-8") as f:
        json.dump(todos_metadados, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Indexação concluída!")
    print(f"   {len(todos_metadados)} páginas indexadas")
    print(f"   Índice salvo em: {pasta_saida}/")
    print(f"   → embeddings.npy  ({matriz.shape})")
    print(f"   → metadados.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa PDFs de desenhos técnicos com CLIP.")
    parser.add_argument("--pasta", required=True, help="Pasta raiz com os PDFs")
    parser.add_argument("--saida", default="./indice", help="Pasta para salvar o índice (default: ./indice)")
    args = parser.parse_args()
    indexar(args.pasta, args.saida)
