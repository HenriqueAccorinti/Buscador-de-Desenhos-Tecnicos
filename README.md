# Busca por Similaridade em Desenhos Técnicos (CLIP)

Dois scripts Python para indexar PDFs de projetos de engenharia e buscar similares
por imagem, PDF ou descrição textual — 100% local, sem custo de API.

---

## Como funciona

```
PDFs → conversão para imagem (PyMuPDF) → embedding CLIP (vetor numérico)
                                                    ↓
Input (PDF/JPG/texto) → embedding CLIP → similaridade de cosseno → ranking
```

O modelo **CLIP ViT-L/14** transforma imagens e textos em vetores de 768 dimensões.
Projetos visualmente parecidos ficam próximos no espaço vetorial.

---

## Instalação

```bash
pip install torch torchvision open_clip_torch pymupdf pillow numpy tqdm
```

> **GPU (opcional):** Se tiver uma GPU NVIDIA, instale a versão CUDA do PyTorch:
> https://pytorch.org/get-started/locally/
> A indexação fica ~10× mais rápida com GPU.

---

## Uso

### Passo 1 — Indexar os PDFs (feito uma vez)

```bash
python indexar.py --pasta /caminho/para/seus/pdfs --saida ./indice
```

Isso gera dois arquivos na pasta `./indice/`:
- `embeddings.npy` — matriz de vetores
- `metadados.json` — mapeamento vetor → arquivo/página

**Tempo estimado:** ~1–3 segundos por PDF (CPU). Com GPU, ~10× mais rápido.

> **Reindexar após adicionar novos PDFs:** rode o mesmo comando novamente.
> O índice é reconstruído do zero.

---

### Passo 2 — Buscar similares

**Por imagem ou PDF:**
```bash
python buscar.py --input meu_projeto.pdf --top 10
python buscar.py --input referencia.jpg  --top 5
```

**Por descrição textual:**
```bash
python buscar.py --texto "planta baixa residencial" --top 10
python buscar.py --texto "diagrama elétrico quadro de distribuição" --top 5
```

---

## Opções de configuração (em indexar.py)

| Variável | Padrão | Descrição |
|---|---|---|
| `DPI` | 150 | Resolução da conversão PDF→imagem. 150 é bom equilíbrio. Aumente para 200–300 em detalhes finos. |
| `MAX_PAGINAS` | 1 | Páginas por PDF a indexar. `1` = só a primeira (capa/folha de rosto). `None` = todas. |
| `BATCH_SIZE` | 16 | Imagens por batch. Reduza para 4–8 se der erro de memória. |
| `MODELO` | ViT-L-14 | Modelo CLIP. Opções maiores (ViT-H-14) são mais precisos porém mais lentos. |

---

## Limitações e dicas

- **CLIP não foi treinado em desenhos técnicos**: funciona bem para distinguir
  categorias grandes (plantas baixas vs. diagramas elétricos vs. peças mecânicas),
  mas pode errar em projetos muito similares que diferem apenas em detalhes numéricos.

- **Busca textual em inglês**: o CLIP tem melhor desempenho com termos em inglês.
  "floor plan" tende a funcionar melhor que "planta baixa".

- **Se a qualidade for insuficiente**: considere fine-tuning do CLIP com pares
  de projetos do seu acervo rotulados como similares/diferentes.

---

## Estrutura de arquivos

```
projeto/
├── indexar.py       ← roda uma vez para criar o índice
├── buscar.py        ← roda sempre que quiser buscar
└── indice/
    ├── embeddings.npy   ← vetores (gerado automaticamente)
    └── metadados.json   ← mapeamento (gerado automaticamente)
```
