"""
app.py — Interface gráfica para busca de desenhos técnicos por similaridade (CLIP).

Uso:
    python app.py

Dependências:
    pip install torch torchvision open_clip_torch pymupdf pillow numpy tqdm
"""

import json
import os
import platform
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, messagebox, ttk

import fitz
import numpy as np
import open_clip
import torch
from PIL import Image, ImageTk

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
MODELO      = "ViT-L-14"
PRETRAINED  = "openai"
DPI         = 150
MAX_PAGINAS = 1
BATCH_SIZE  = 16
PREVIEW_W   = 200   # largura dos thumbnails de resultado
PREVIEW_H   = 240   # altura dos thumbnails de resultado

DARK        = "#0f0f10"
PANEL       = "#1a1a1e"
SURFACE     = "#222228"
BORDER      = "#2e2e38"
ACCENT      = "#4f7cff"
ACCENT2     = "#7c4fff"
TEXT        = "#e8e8f0"
MUTED       = "#6b6b7e"
SUCCESS     = "#3ecf8e"
WARNING     = "#f5a623"
ERROR       = "#f54e42"

# ─────────────────────────────────────────────────────────────────────────────
# Lógica de ML (igual aos scripts originais)
# ─────────────────────────────────────────────────────────────────────────────

def carregar_modelo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODELO, pretrained=PRETRAINED)
    model = model.to(device).eval()
    return model, preprocess, device


def pdf_para_imagens(caminho_pdf: Path, max_paginas=MAX_PAGINAS, dpi=DPI):
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
    except Exception:
        pass
    return imagens


def calcular_embeddings_lote(imgs, model, preprocess, device):
    tensors = torch.stack([preprocess(img) for img in imgs]).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensors)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


def embedding_imagem_unica(pil_img, model, preprocess, device):
    t = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


def carregar_imagem_input(caminho: str):
    path = Path(caminho)
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        page = doc[0]
        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    else:
        return Image.open(str(path)).convert("RGB")


def gerar_thumbnail(pil_img, w=PREVIEW_W, h=PREVIEW_H):
    """Gera thumbnail com aspect ratio preservado e fundo escuro."""
    img = pil_img.copy()
    img.thumbnail((w, h - 40), Image.LANCZOS)
    bg = Image.new("RGB", (w, h - 40), color=(30, 30, 38))
    offset = ((w - img.width) // 2, (h - 40 - img.height) // 2)
    bg.paste(img, offset)
    return bg


def abrir_pasta_do_arquivo(caminho_arquivo: str):
    """Abre o explorador de arquivos na pasta do arquivo, selecionando-o.
    Reutiliza janela já aberta no Windows; no Mac/Linux abre o Finder/Nautilus."""
    path = Path(caminho_arquivo).resolve()
    sistema = platform.system()
    try:
        if sistema == "Windows":
            # /select abre o Explorer e seleciona o arquivo; se já estiver aberto, foca
            subprocess.Popen(["explorer", "/select,", str(path)])
        elif sistema == "Darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            # Linux — abre a pasta (Nautilus, Dolphin, etc.)
            subprocess.Popen(["xdg-open", str(path.parent)])
    except Exception as e:
        messagebox.showerror("Erro", f"Não foi possível abrir o explorador:\n{e}")


# ─────────────────────────────────────────────────────────────────────────────
# Widgets customizados
# ─────────────────────────────────────────────────────────────────────────────

class FlatButton(tk.Label):
    """Botão flat estilizado (Label clicável)."""

    def __init__(self, parent, text, command=None, accent=False,
                 small=False, danger=False, **kwargs):
        bg = ACCENT if accent else (ERROR if danger else SURFACE)
        fg = "#ffffff" if (accent or danger) else TEXT
        pad_x = 12 if small else 18
        pad_y = 5 if small else 9

        super().__init__(
            parent, text=text, bg=bg, fg=fg,
            cursor="hand2", padx=pad_x, pady=pad_y,
            relief="flat", **kwargs
        )
        self._bg_normal = bg
        self._bg_hover  = ACCENT2 if accent else (BORDER if not danger else "#c93c32")
        self._command   = command
        self.bind("<Enter>",    lambda e: self.config(bg=self._bg_hover))
        self.bind("<Leave>",    lambda e: self.config(bg=self._bg_normal))
        self.bind("<Button-1>", lambda e: command() if command else None)


class ScrollableFrame(tk.Frame):
    """Frame com scrollbar vertical."""

    def __init__(self, parent, **kwargs):
        outer = tk.Frame(parent, bg=DARK)
        outer.pack(fill="both", expand=True)

        canvas_bg = kwargs.pop("bg", DARK)

        self._canvas = tk.Canvas(outer, bg=canvas_bg, highlightthickness=0,
                                 bd=0)
        sb = tk.Scrollbar(outer, orient="vertical",
                          command=self._canvas.yview,
                          bg=SURFACE, troughcolor=DARK,
                          activebackground=ACCENT)
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        super().__init__(self._canvas, bg=canvas_bg)
        self._window = self._canvas.create_window((0, 0), window=self,
                                                  anchor="nw")
        self.bind("<Configure>", self._on_frame_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_frame_configure(self, _):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self._canvas.itemconfig(self._window, width=event.width)

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ─────────────────────────────────────────────────────────────────────────────
# Aplicação principal
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("DrawSearch — Busca de Desenhos Técnicos")
        self.configure(bg=DARK)
        self.geometry("1100x780")
        self.minsize(900, 600)

        # Estado
        self.model        = None
        self.preprocess   = None
        self.device       = None
        self.embeddings   = None   # np.ndarray (N, D)
        self.metadados    = None   # list[dict]
        self._model_lock  = threading.Lock()
        self._thumb_cache = {}     # caminho → ImageTk

        # Variáveis de controle
        self.var_pasta_pdfs  = tk.StringVar()
        self.var_pasta_indice = tk.StringVar(value=str(Path.home() / "drawsearch_indice"))
        self.var_input        = tk.StringVar()
        self.var_top          = tk.IntVar(value=12)
        self.var_status       = tk.StringVar(value="Pronto.")
        self.var_progresso    = tk.DoubleVar(value=0)

        self._build_ui()
        self._carregar_modelo_background()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        paned = tk.PanedWindow(self, orient="horizontal",
                               bg=DARK, sashwidth=4, sashrelief="flat",
                               sashpad=0)
        paned.pack(fill="both", expand=True, padx=0, pady=0)

        left = self._build_left_panel(paned)
        right = self._build_right_panel(paned)

        paned.add(left,  minsize=320, width=360)
        paned.add(right, minsize=500)

        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=PANEL, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⬡", bg=PANEL, fg=ACCENT,
                 font=("Courier", 22)).pack(side="left", padx=(18, 6), pady=8)
        tk.Label(hdr, text="DrawSearch", bg=PANEL, fg=TEXT,
                 font=("Courier", 16, "bold")).pack(side="left", pady=8)
        tk.Label(hdr, text="  busca por similaridade visual em desenhos técnicos",
                 bg=PANEL, fg=MUTED,
                 font=("Courier", 9)).pack(side="left", pady=8)

        # Tag de dispositivo (CPU/GPU)
        self.lbl_device = tk.Label(hdr, text="carregando…", bg=PANEL,
                                   fg=MUTED, font=("Courier", 9))
        self.lbl_device.pack(side="right", padx=18)

    def _build_left_panel(self, parent):
        frame = tk.Frame(parent, bg=PANEL, bd=0)

        # ── Seção: Indexar ───────────────────────────────────────────────────
        self._section(frame, "01  INDEXAR ACERVO")

        row_pasta = tk.Frame(frame, bg=PANEL)
        row_pasta.pack(fill="x", padx=18, pady=(0, 4))
        tk.Label(row_pasta, text="Pasta de PDFs", bg=PANEL, fg=MUTED,
                 font=("Courier", 8)).pack(anchor="w")
        row_e1 = tk.Frame(row_pasta, bg=PANEL)
        row_e1.pack(fill="x")
        entry_pasta = tk.Entry(row_e1, textvariable=self.var_pasta_pdfs,
                               bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                               relief="flat", font=("Courier", 10),
                               bd=0, highlightthickness=1,
                               highlightbackground=BORDER,
                               highlightcolor=ACCENT)
        entry_pasta.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 6))
        FlatButton(row_e1, "…", command=self._escolher_pasta_pdfs,
                   small=True).pack(side="left")

        row_indice = tk.Frame(frame, bg=PANEL)
        row_indice.pack(fill="x", padx=18, pady=(0, 10))
        tk.Label(row_indice, text="Salvar índice em", bg=PANEL, fg=MUTED,
                 font=("Courier", 8)).pack(anchor="w")
        row_e2 = tk.Frame(row_indice, bg=PANEL)
        row_e2.pack(fill="x")
        tk.Entry(row_e2, textvariable=self.var_pasta_indice,
                 bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Courier", 10),
                 bd=0, highlightthickness=1,
                 highlightbackground=BORDER,
                 highlightcolor=ACCENT).pack(side="left", fill="x", expand=True,
                                             ipady=6, padx=(0, 6))
        FlatButton(row_e2, "…", command=self._escolher_pasta_indice,
                   small=True).pack(side="left")

        # Barra de progresso
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("DS.Horizontal.TProgressbar",
                        troughcolor=SURFACE, background=ACCENT,
                        darkcolor=ACCENT, lightcolor=ACCENT,
                        bordercolor=PANEL, thickness=4)
        self.progressbar = ttk.Progressbar(frame, variable=self.var_progresso,
                                           maximum=100, mode="determinate",
                                           style="DS.Horizontal.TProgressbar")
        self.progressbar.pack(fill="x", padx=18, pady=(0, 8))

        btn_row = tk.Frame(frame, bg=PANEL)
        btn_row.pack(fill="x", padx=18, pady=(0, 16))
        FlatButton(btn_row, "▶  Iniciar Indexação",
                   command=self._iniciar_indexacao, accent=True).pack(fill="x")

        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", padx=18, pady=4)

        # ── Seção: Buscar ────────────────────────────────────────────────────
        self._section(frame, "02  BUSCAR")

        row_inp = tk.Frame(frame, bg=PANEL)
        row_inp.pack(fill="x", padx=18, pady=(0, 4))
        tk.Label(row_inp, text="Arquivo de busca  (PDF, JPG, PNG)",
                 bg=PANEL, fg=MUTED, font=("Courier", 8)).pack(anchor="w")
        row_e3 = tk.Frame(row_inp, bg=PANEL)
        row_e3.pack(fill="x")
        tk.Entry(row_e3, textvariable=self.var_input,
                 bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Courier", 10),
                 bd=0, highlightthickness=1,
                 highlightbackground=BORDER,
                 highlightcolor=ACCENT).pack(side="left", fill="x", expand=True,
                                             ipady=6, padx=(0, 6))
        FlatButton(row_e3, "…", command=self._escolher_input,
                   small=True).pack(side="left")

        row_top = tk.Frame(frame, bg=PANEL)
        row_top.pack(fill="x", padx=18, pady=(0, 10))
        tk.Label(row_top, text="Resultados a exibir", bg=PANEL, fg=MUTED,
                 font=("Courier", 8)).pack(anchor="w")
        scale = tk.Scale(row_top, variable=self.var_top, from_=3, to=30,
                         orient="horizontal", bg=PANEL, fg=TEXT,
                         troughcolor=SURFACE, activebackground=ACCENT,
                         highlightthickness=0, sliderrelief="flat",
                         bd=0, font=("Courier", 9))
        scale.pack(fill="x")

        btn_row2 = tk.Frame(frame, bg=PANEL)
        btn_row2.pack(fill="x", padx=18, pady=(0, 16))
        FlatButton(btn_row2, "⌕  Buscar Similares",
                   command=self._iniciar_busca, accent=True).pack(fill="x")

        tk.Frame(frame, bg=BORDER, height=1).pack(fill="x", padx=18, pady=4)

        # ── Seção: Índice carregado ──────────────────────────────────────────
        self._section(frame, "03  ÍNDICE ATUAL")
        self.lbl_indice_info = tk.Label(frame, text="Nenhum índice carregado.",
                                        bg=PANEL, fg=MUTED,
                                        font=("Courier", 9),
                                        justify="left", wraplength=300)
        self.lbl_indice_info.pack(padx=18, pady=(0, 8), anchor="w")
        FlatButton(frame, "Carregar Índice Existente",
                   command=self._carregar_indice, small=True).pack(padx=18, anchor="w")

        return frame

    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=DARK)

        # Cabeçalho dos resultados
        hdr = tk.Frame(frame, bg=DARK)
        hdr.pack(fill="x", padx=18, pady=(12, 6))
        self.lbl_resultados = tk.Label(hdr, text="Resultados", bg=DARK,
                                       fg=MUTED, font=("Courier", 11))
        self.lbl_resultados.pack(side="left")

        # Preview do input
        self.frame_preview_input = tk.Frame(hdr, bg=DARK)
        self.frame_preview_input.pack(side="right")
        self.lbl_input_thumb = tk.Label(self.frame_preview_input, bg=DARK)
        self.lbl_input_thumb.pack(side="right")
        tk.Label(self.frame_preview_input, text="busca: ", bg=DARK,
                 fg=MUTED, font=("Courier", 8)).pack(side="right")

        # Área de resultados com scroll
        self.scroll_frame = ScrollableFrame(frame, bg=DARK)
        self.scroll_frame.pack(fill="both", expand=True)

        # Placeholder
        self.lbl_placeholder = tk.Label(self.scroll_frame,
                                        text="Indexe um acervo e escolha um arquivo para buscar.",
                                        bg=DARK, fg=MUTED, font=("Courier", 11))
        self.lbl_placeholder.pack(pady=80)

        return frame

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=SURFACE, height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self.var_status, bg=SURFACE, fg=MUTED,
                 font=("Courier", 9), anchor="w").pack(side="left", padx=12)

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", padx=18, pady=(16, 6))
        tk.Label(f, text=title, bg=PANEL, fg=ACCENT,
                 font=("Courier", 9, "bold")).pack(anchor="w")

    # ── Seleção de arquivos/pastas ──────────────────────────────────────────

    def _escolher_pasta_pdfs(self):
        p = filedialog.askdirectory(title="Selecione a pasta com os PDFs")
        if p:
            self.var_pasta_pdfs.set(p)

    def _escolher_pasta_indice(self):
        p = filedialog.askdirectory(title="Selecione onde salvar o índice")
        if p:
            self.var_pasta_indice.set(p)

    def _escolher_input(self):
        p = filedialog.askopenfilename(
            title="Selecione o arquivo de busca",
            filetypes=[("Imagens e PDFs",
                        "*.pdf *.PDF *.jpg *.jpeg *.png *.PNG *.JPG *.JPEG")])
        if p:
            self.var_input.set(p)
            self._mostrar_thumb_input(p)

    def _mostrar_thumb_input(self, caminho):
        try:
            img = carregar_imagem_input(caminho)
            img.thumbnail((60, 60), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            self.lbl_input_thumb.config(image=tk_img)
            self.lbl_input_thumb._img = tk_img  # evita GC
        except Exception:
            pass

    # ── Carregamento de modelo (background) ────────────────────────────────

    def _carregar_modelo_background(self):
        self.var_status.set("Carregando modelo CLIP… (pode demorar na primeira vez)")

        def _worker():
            try:
                model, preprocess, device = carregar_modelo()
                with self._model_lock:
                    self.model      = model
                    self.preprocess = preprocess
                    self.device     = device
                self.after(0, lambda: self.lbl_device.config(
                    text=f"✓ CLIP  |  {device.upper()}", fg=SUCCESS))
                self.after(0, lambda: self.var_status.set("Modelo carregado. Pronto."))
            except Exception as e:
                self.after(0, lambda: self.var_status.set(f"Erro ao carregar modelo: {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    # ── Indexação ───────────────────────────────────────────────────────────

    def _iniciar_indexacao(self):
        pasta = self.var_pasta_pdfs.get().strip()
        saida  = self.var_pasta_indice.get().strip()

        if not pasta or not Path(pasta).is_dir():
            messagebox.showerror("Erro", "Selecione uma pasta de PDFs válida.")
            return
        if not saida:
            messagebox.showerror("Erro", "Defina a pasta de destino do índice.")
            return
        if self.model is None:
            messagebox.showwarning("Aguarde", "O modelo ainda está carregando.")
            return

        threading.Thread(target=self._indexar_worker,
                         args=(pasta, saida), daemon=True).start()

    def _indexar_worker(self, pasta_pdfs, pasta_saida):
        self.after(0, lambda: self.var_status.set("Iniciando indexação…"))
        self.after(0, lambda: self.var_progresso.set(0))

        Path(pasta_saida).mkdir(parents=True, exist_ok=True)

        # rglob captura PDFs em qualquer subpasta
        pdfs = sorted(Path(pasta_pdfs).rglob("*.pdf")) + \
               sorted(Path(pasta_pdfs).rglob("*.PDF"))

        if not pdfs:
            self.after(0, lambda: messagebox.showwarning(
                "Aviso", f"Nenhum PDF encontrado em:\n{pasta_pdfs}"))
            return

        total = len(pdfs)
        todos_emb  = []
        todos_meta = []

        for i, pdf_path in enumerate(pdfs):
            pct = int((i / total) * 100)
            nome = pdf_path.name
            self.after(0, lambda p=pct, n=nome:
                       (self.var_progresso.set(p),
                        self.var_status.set(f"[{p}%]  {n}")))

            paginas = pdf_para_imagens(pdf_path)
            if not paginas:
                continue

            for inicio in range(0, len(paginas), BATCH_SIZE):
                lote = paginas[inicio:inicio + BATCH_SIZE]
                idx_pags = [x[0] for x in lote]
                imgs     = [x[1] for x in lote]
                with self._model_lock:
                    embs = calcular_embeddings_lote(
                        imgs, self.model, self.preprocess, self.device)
                for emb, pag in zip(embs, idx_pags):
                    todos_emb.append(emb)
                    todos_meta.append({
                        "arquivo": str(pdf_path.relative_to(pasta_pdfs)),
                        "caminho_completo": str(pdf_path),
                        "pagina": pag,
                    })

        if not todos_emb:
            self.after(0, lambda: messagebox.showerror(
                "Erro", "Nenhum embedding gerado."))
            return

        matriz = np.stack(todos_emb)
        np.save(str(Path(pasta_saida) / "embeddings.npy"), matriz)
        with open(Path(pasta_saida) / "metadados.json", "w", encoding="utf-8") as f:
            json.dump(todos_meta, f, ensure_ascii=False, indent=2)

        # Carrega o índice recém-criado
        self.embeddings = matriz
        self.metadados  = todos_meta

        n = len(todos_meta)
        self.after(0, lambda: (
            self.var_progresso.set(100),
            self.var_status.set(f"✓ Indexação concluída — {n} páginas indexadas."),
            self.lbl_indice_info.config(
                text=f"{n} páginas indexadas\n{total} PDFs  |  {pasta_saida}",
                fg=SUCCESS),
            messagebox.showinfo("Concluído",
                                f"Indexação concluída!\n{n} páginas de {total} PDFs.")
        ))

    # ── Carregamento de índice existente ────────────────────────────────────

    def _carregar_indice(self):
        pasta = filedialog.askdirectory(title="Selecione a pasta do índice")
        if not pasta:
            return
        emb_path  = Path(pasta) / "embeddings.npy"
        meta_path = Path(pasta) / "metadados.json"
        if not emb_path.exists() or not meta_path.exists():
            messagebox.showerror("Erro",
                                 "Pasta inválida: não contém embeddings.npy e metadados.json")
            return
        try:
            self.embeddings = np.load(str(emb_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadados = json.load(f)
            n = len(self.metadados)
            self.lbl_indice_info.config(
                text=f"{n} páginas carregadas\n{pasta}", fg=SUCCESS)
            self.var_status.set(f"Índice carregado: {n} páginas.")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar índice:\n{e}")

    # ── Busca ───────────────────────────────────────────────────────────────

    def _iniciar_busca(self):
        if self.embeddings is None or self.metadados is None:
            messagebox.showwarning("Aviso",
                                   "Nenhum índice carregado.\nIndexe um acervo ou carregue um índice existente.")
            return
        inp = self.var_input.get().strip()
        if not inp or not Path(inp).exists():
            messagebox.showerror("Erro",
                                 "Selecione um arquivo de busca válido (PDF, JPG, PNG).")
            return
        if self.model is None:
            messagebox.showwarning("Aguarde", "O modelo ainda está carregando.")
            return

        self.var_status.set("Buscando…")
        threading.Thread(target=self._busca_worker, args=(inp,), daemon=True).start()

    def _busca_worker(self, input_path):
        try:
            img = carregar_imagem_input(input_path)
            with self._model_lock:
                query = embedding_imagem_unica(
                    img, self.model, self.preprocess, self.device)

            scores = (self.embeddings @ query.T).flatten()
            ranking = np.argsort(scores)[::-1]

            # Deduplica por arquivo
            vistos = {}
            for idx in ranking:
                arq = self.metadados[idx]["arquivo"]
                if arq not in vistos:
                    vistos[arq] = (idx, float(scores[idx]))
                if len(vistos) >= self.var_top.get():
                    break

            resultados = [
                {
                    "rank": i + 1,
                    "score": score,
                    "arquivo": arq,
                    "pagina": self.metadados[idx]["pagina"] + 1,
                    "caminho_completo": self.metadados[idx]["caminho_completo"],
                }
                for i, (arq, (idx, score)) in enumerate(vistos.items())
            ]

            self.after(0, lambda: self._exibir_resultados(resultados))
        except Exception as e:
            self.after(0, lambda: (
                self.var_status.set(f"Erro na busca: {e}"),
                messagebox.showerror("Erro", f"Falha na busca:\n{e}")
            ))

    # ── Exibição de resultados ──────────────────────────────────────────────

    def _exibir_resultados(self, resultados):
        # Limpa área anterior
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self._thumb_cache.clear()

        n = len(resultados)
        self.lbl_resultados.config(
            text=f"{n} resultado{'s' if n != 1 else ''} encontrado{'s' if n != 1 else ''}",
            fg=TEXT)
        self.var_status.set(f"Busca concluída — {n} resultado(s).")

        # Grade responsiva: 4 colunas fixas
        COLS = 4
        grid = tk.Frame(self.scroll_frame, bg=DARK)
        grid.pack(fill="both", expand=True, padx=12, pady=12)

        for r in resultados:
            col = (r["rank"] - 1) % COLS
            row = (r["rank"] - 1) // COLS
            self._criar_card(grid, r, row, col)

    def _criar_card(self, parent, resultado, row, col):
        card = tk.Frame(parent, bg=SURFACE, bd=0,
                        highlightthickness=1, highlightbackground=BORDER)
        card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
        parent.columnconfigure(col, weight=1)

        # Thumbnail
        thumb_lbl = tk.Label(card, bg=SURFACE, cursor="hand2")
        thumb_lbl.pack(fill="x")
        self._carregar_thumbnail_async(resultado["caminho_completo"], thumb_lbl)

        # Score bar
        score = resultado["score"]
        bar_frame = tk.Frame(card, bg=SURFACE)
        bar_frame.pack(fill="x", padx=8, pady=(4, 0))
        bar_w = PREVIEW_W - 16
        filled = max(4, int(score * bar_w))
        tk.Frame(bar_frame, bg=ACCENT, width=filled, height=3).pack(side="left")
        tk.Frame(bar_frame, bg=BORDER,
                 width=bar_w - filled, height=3).pack(side="left")

        # Info
        info = tk.Frame(card, bg=SURFACE)
        info.pack(fill="x", padx=8, pady=(4, 2))

        nome = Path(resultado["arquivo"]).name
        if len(nome) > 28:
            nome = nome[:25] + "…"

        tk.Label(info, text=nome, bg=SURFACE, fg=TEXT,
                 font=("Courier", 8, "bold"),
                 wraplength=PREVIEW_W - 16, justify="left",
                 anchor="w").pack(anchor="w")

        subpasta = str(Path(resultado["arquivo"]).parent)
        if subpasta != ".":
            tk.Label(info, text=subpasta, bg=SURFACE, fg=MUTED,
                     font=("Courier", 7),
                     wraplength=PREVIEW_W - 16, justify="left",
                     anchor="w").pack(anchor="w")

        tk.Label(info, text=f"similaridade: {score:.1%}",
                 bg=SURFACE, fg=ACCENT,
                 font=("Courier", 8)).pack(anchor="w")

        # Botão abrir
        btn_frame = tk.Frame(card, bg=SURFACE)
        btn_frame.pack(fill="x", padx=8, pady=(2, 8))
        caminho = resultado["caminho_completo"]
        FlatButton(btn_frame, "📂  Abrir no Explorer",
                   command=lambda c=caminho: abrir_pasta_do_arquivo(c),
                   small=True).pack(fill="x")

    def _carregar_thumbnail_async(self, caminho_pdf, label):
        """Carrega o thumbnail em background para não travar a UI."""
        if caminho_pdf in self._thumb_cache:
            label.config(image=self._thumb_cache[caminho_pdf])
            return

        def _worker():
            try:
                paginas = pdf_para_imagens(Path(caminho_pdf), max_paginas=1, dpi=96)
                if not paginas:
                    raise ValueError("sem páginas")
                img = gerar_thumbnail(paginas[0][1], PREVIEW_W, PREVIEW_H)
                tk_img = ImageTk.PhotoImage(img)
                self._thumb_cache[caminho_pdf] = tk_img
                self.after(0, lambda lbl=label, im=tk_img:
                           lbl.config(image=im, width=PREVIEW_W, height=PREVIEW_H - 40))
            except Exception:
                # Placeholder cinza
                placeholder = Image.new("RGB", (PREVIEW_W, PREVIEW_H - 40), (40, 40, 50))
                tk_img = ImageTk.PhotoImage(placeholder)
                self._thumb_cache[caminho_pdf] = tk_img
                self.after(0, lambda lbl=label, im=tk_img: lbl.config(image=im))

        threading.Thread(target=_worker, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
