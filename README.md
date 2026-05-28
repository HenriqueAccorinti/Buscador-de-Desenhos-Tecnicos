# DrawSearch — Busca por Similaridade em Desenhos Técnicos

Ferramenta local para indexar acervos de PDFs de engenharia e encontrar projetos
visualmente similares a um arquivo de referência — sem custo de API, sem internet.

---

## Como funciona

```
PDFs → conversão para imagem (PyMuPDF) → embedding CLIP (vetor numérico)
                                                    ↓
Input (PDF/JPG/PNG) → embedding CLIP → similaridade de cosseno → ranking com previews
```

O modelo **CLIP ViT-L/14** transforma imagens em vetores de 768 dimensões.
Projetos visualmente parecidos ficam próximos nesse espaço vetorial.

---

## Instalação

```bash
pip install torch torchvision open_clip_torch pymupdf pillow numpy tqdm
```

> **GPU (opcional):** Se tiver uma GPU NVIDIA, instale a versão CUDA do PyTorch:
> https://pytorch.org/get-started/locally/
> A indexação fica ~10× mais rápida com GPU.

---

## Uso — Interface gráfica

```bash
python app.py
```

### Fluxo básico

**1. Carregar ou criar um índice**

- Se já tiver um índice gerado anteriormente, clique em **Carregar Índice Existente**
  e selecione a pasta onde ele foi salvo.
- Para criar um novo índice, abra a seção **CONFIGURAÇÕES** (clique no título para expandir),
  selecione a pasta raiz dos seus PDFs e clique em **Iniciar Indexação**.
  Todos os PDFs dentro da pasta e subpastas serão indexados automaticamente.

**2. Buscar projetos similares**

- Clique em **"…"** ao lado do campo de busca e selecione um PDF, JPG ou PNG de referência.
- Ajuste o slider para definir quantos resultados exibir.
- Clique em **Buscar Similares**.

**3. Ver resultados**

- Os resultados aparecem em grade com thumbnails, nome do arquivo, subpasta e percentual de similaridade.
- **Clique no thumbnail** para ampliar a imagem em tela cheia. Feche com Esc ou clique.
- **Clique em "Abrir no Explorer"** para abrir o explorador de arquivos com o arquivo selecionado.

---

## Uso — Scripts de linha de comando

Os scripts `indexar.py` e `buscar.py` funcionam de forma independente, sem interface gráfica.

### Indexar

```bash
python indexar.py --pasta /caminho/para/seus/pdfs --saida ./indice
```

### Buscar

```bash
python buscar.py --input meu_projeto.pdf --top 10
python buscar.py --input referencia.jpg  --top 5
```

---

## Opções de configuração

As constantes abaixo ficam no topo de `app.py` (e de `indexar.py` / `buscar.py`):

| Variável | Padrão | Descrição |
|---|---|---|
| `DPI` | 150 | Resolução da conversão PDF→imagem. Aumente para 200–300 para detalhes finos. |
| `MAX_PAGINAS` | 1 | Páginas por PDF a indexar. `1` = só a primeira. `None` = todas. |
| `BATCH_SIZE` | 16 | Imagens por batch. Reduza para 4–8 se der erro de memória. |
| `MODELO` | ViT-L-14 | Modelo CLIP. `ViT-H-14` é mais preciso, porém mais lento. |
| `PREVIEW_W/H` | 200 | Tamanho dos thumbnails na grade de resultados. |

---

## Limitações

- **CLIP não foi treinado especificamente em desenhos técnicos:** funciona bem para
  distinguir categorias amplas (plantas baixas, diagramas elétricos, peças mecânicas),
  mas pode errar em projetos muito similares que diferem apenas em detalhes numéricos.
- **Reindexação necessária** ao adicionar novos PDFs ao acervo: basta rodar a indexação
  novamente apontando para a mesma pasta.

---

## Estrutura de arquivos

```
projeto/
├── app.py           ← interface gráfica (uso principal)
├── indexar.py       ← indexação via linha de comando
├── buscar.py        ← busca via linha de comando
├── .gitignore
└── indice/          ← gerado automaticamente (não sobe para o git)
    ├── embeddings.npy
    └── metadados.json
```