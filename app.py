"""
app.py — Interface gráfica para busca de desenhos técnicos por similaridade (CLIP).

Uso:
    python app.py

Dependências:
    pip install torch torchvision open_clip_torch pymupdf pillow numpy tqdm
"""

import json
import platform
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

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
PREVIEW_W   = 200
PREVIEW_H   = 200

DARK    = "#0f0f10"
PANEL   = "#1a1a1e"
SURFACE = "#222228"
BORDER  = "#2e2e38"
ACCENT  = "#4f7cff"
ACCENT2 = "#7c4fff"
TEXT    = "#e8e8f0"
MUTED   = "#6b6b7e"
SUCCESS = "#3ecf8e"
ERROR   = "#f54e42"

# ─────────────────────────────────────────────────────────────────────────────
# Lógica ML
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
    img = pil_img.copy()
    img.thumbnail((w, h), Image.LANCZOS)
    bg = Image.new("RGB", (w, h), color=(30, 30, 38))
    offset = ((w - img.width) // 2, (h - img.height) // 2)
    bg.paste(img, offset)
    return bg


def abrir_pasta_do_arquivo(caminho_arquivo: str):
    """Abre o explorador selecionando o arquivo.
    No Windows, explorer /select reutiliza a janela já aberta se a pasta for a mesma."""
    path = Path(caminho_arquivo).resolve()
    sistema = platform.system()
    try:
        if sistema == "Windows":
            subprocess.Popen(["explorer", "/select,", str(path)])
        elif sistema == "Darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)])
    except Exception as e:
        messagebox.showerror("Erro", f"Não foi possível abrir o explorador:\n{e}")


# ─────────────────────────────────────────────────────────────────────────────
# Widgets
# ─────────────────────────────────────────────────────────────────────────────

class FlatButton(tk.Label):
    def __init__(self, parent, text, command=None, accent=False,
                 small=False, danger=False, **kwargs):
        bg = ACCENT if accent else (ERROR if danger else SURFACE)
        fg = "#ffffff" if (accent or danger) else TEXT
        pad_x = 12 if small else 18
        pad_y = 5 if small else 9
        super().__init__(parent, text=text, bg=bg, fg=fg,
                         cursor="hand2", padx=pad_x, pady=pad_y,
                         relief="flat", **kwargs)
        self._bg_normal = bg
        self._bg_hover  = ACCENT2 if accent else (BORDER if not danger else "#c93c32")
        self.bind("<Enter>",    lambda e: self.config(bg=self._bg_hover))
        self.bind("<Leave>",    lambda e: self.config(bg=self._bg_normal))
        self.bind("<Button-1>", lambda e: command() if command else None)


class CollapseSection(tk.Frame):
    """Seção recolhível: clique no título abre/fecha o conteúdo."""

    def __init__(self, parent, title, bg=PANEL, **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        self._open = False
        self._bg   = bg

        # Header clicável
        hdr = tk.Frame(self, bg=bg, cursor="hand2")
        hdr.pack(fill="x", padx=18, pady=(12, 0))

        self._arrow = tk.Label(hdr, text="▶", bg=bg, fg=ACCENT,
                               font=("Courier", 9), cursor="hand2")
        self._arrow.pack(side="left", padx=(0, 6))
        tk.Label(hdr, text=title, bg=bg, fg=ACCENT,
                 font=("Courier", 9, "bold"), cursor="hand2").pack(side="left")

        hdr.bind("<Button-1>", self._toggle)
        self._arrow.bind("<Button-1>", self._toggle)

        # Corpo (inicialmente oculto)
        self.body = tk.Frame(self, bg=bg)
        # não empacota ainda

    def _toggle(self, _=None):
        self._open = not self._open
        if self._open:
            self.body.pack(fill="x")
            self._arrow.config(text="▼")
        else:
            self.body.forget()
            self._arrow.config(text="▶")


class ScrollableResults(tk.Frame):
    """
    Frame com Canvas + Scrollbar vertical.
    O scroll funciona tanto com a barra lateral quanto com o scroll do mouse
    (inclusive em subwidgets filhos).
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=DARK, **kwargs)

        self._canvas = tk.Canvas(self, bg=DARK, highlightthickness=0, bd=0)
        sb = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview,
                          bg=SURFACE, troughcolor=DARK, activebackground=ACCENT)
        self._canvas.configure(yscrollcommand=sb.set)

        sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        # Frame interno onde colocamos os cards
        self.inner = tk.Frame(self._canvas, bg=DARK)
        self._win  = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._update_scrollregion)
        self._canvas.bind("<Configure>", self._update_inner_width)

        # Bind scroll do mouse no canvas E em qualquer widget filho
        self._canvas.bind("<MouseWheel>",     self._on_scroll)
        self._canvas.bind("<Button-4>",       self._on_scroll)   # Linux scroll up
        self._canvas.bind("<Button-5>",       self._on_scroll)   # Linux scroll down
        self.bind_all("<MouseWheel>",         self._on_scroll)
        self.bind_all("<Button-4>",           self._on_scroll)
        self.bind_all("<Button-5>",           self._on_scroll)

    def _update_scrollregion(self, _=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _update_inner_width(self, event):
        self._canvas.itemconfig(self._win, width=event.width)

    def _on_scroll(self, event):
        # Windows/Mac: event.delta  |  Linux: event.num
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5 or (hasattr(event, "delta") and event.delta < 0):
            self._canvas.yview_scroll(1, "units")

    def scroll_to_top(self):
        self._canvas.yview_moveto(0)


# ─────────────────────────────────────────────────────────────────────────────
# Lightbox (visualização em tela cheia)
# ─────────────────────────────────────────────────────────────────────────────

class Lightbox(tk.Toplevel):
    """Exibe uma imagem PIL ocupando quase toda a tela. Fecha com Esc ou clique."""

    def __init__(self, parent, pil_img: Image.Image, titulo: str = ""):
        super().__init__(parent)
        self.configure(bg="#000000")
        self.overrideredirect(True)   # sem barra de título do OS

        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{sw}x{sh}+0+0")
        self.lift()
        self.focus_set()

        # Escurece fundo
        tk.Label(self, bg="#000000").place(relwidth=1, relheight=1)

        # Redimensiona imagem para caber na tela (margem de 5%)
        max_w = int(sw * 0.92)
        max_h = int(sh * 0.92)
        img = pil_img.copy()
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img)

        # Imagem centralizada
        lbl = tk.Label(self, image=self._tk_img, bg="#000000", cursor="hand2")
        lbl.place(relx=0.5, rely=0.5, anchor="center")

        # Nome do arquivo
        if titulo:
            tk.Label(self, text=titulo, bg="#000000", fg="#888888",
                     font=("Courier", 9)).place(relx=0.5, rely=0.97, anchor="center")

        # Botão fechar
        tk.Label(self, text="✕  fechar", bg="#111111", fg="#888888",
                 font=("Courier", 10), cursor="hand2", padx=10, pady=4
                 ).place(relx=0.98, rely=0.02, anchor="ne")

        # Fecha com qualquer clique ou Esc
        self.bind("<Escape>",   lambda _: self.destroy())
        self.bind("<Button-1>", lambda _: self.destroy())
        lbl.bind("<Button-1>",  lambda _: self.destroy())


# ─────────────────────────────────────────────────────────────────────────────
# Aplicação
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("DrawSearch — Busca de Desenhos Técnicos")
        self.configure(bg=DARK)
        self.geometry("1150x800")
        self.minsize(900, 600)

        self.model       = None
        self.preprocess  = None
        self.device      = None
        self.embeddings  = None
        self.metadados   = None
        self._model_lock = threading.Lock()
        self._thumb_refs = {}   # caminho → PIL.Image (full res, para lightbox)
        self._thumb_tk   = {}   # caminho → ImageTk (thumbnail)

        self.var_pasta_pdfs   = tk.StringVar()
        self.var_pasta_indice = tk.StringVar(value=str(Path.home() / "drawsearch_indice"))
        self.var_input        = tk.StringVar()
        self.var_top          = tk.IntVar(value=12)
        self.var_status       = tk.StringVar(value="Carregando modelo de IA…")
        self.var_progresso    = tk.DoubleVar(value=0)

        self._input_pil = None   # PIL da imagem de busca (para lightbox)

        self._build_ui()
        self._carregar_modelo_background()

    # ── Construção da UI ────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()

        paned = tk.PanedWindow(self, orient="horizontal",
                               bg=DARK, sashwidth=5, sashrelief="flat")
        paned.pack(fill="both", expand=True)

        left  = self._build_left_panel(paned)
        right = self._build_right_panel(paned)

        paned.add(left,  minsize=300, width=350)
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
                 bg=PANEL, fg=MUTED, font=("Courier", 9)).pack(side="left", pady=8)

        # Status do modelo — discreto, no canto direito
        self.lbl_model_status = tk.Label(hdr, text="⏳ carregando modelo…",
                                         bg=PANEL, fg=MUTED, font=("Courier", 8))
        self.lbl_model_status.pack(side="right", padx=18)

    def _build_left_panel(self, parent):
        outer = tk.Frame(parent, bg=PANEL)

        # ── Seção de busca (sempre visível) ─────────────────────────────────
        sec_busca = tk.Frame(outer, bg=PANEL)
        sec_busca.pack(fill="x", padx=0, pady=(8, 0))

        tk.Frame(sec_busca, bg=PANEL).pack(fill="x", padx=18, pady=(8, 4))
        tk.Label(sec_busca, text="BUSCAR", bg=PANEL, fg=ACCENT,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=18)

        row_inp = tk.Frame(sec_busca, bg=PANEL)
        row_inp.pack(fill="x", padx=18, pady=(6, 2))
        tk.Label(row_inp, text="Arquivo de busca  (PDF, JPG, PNG)",
                 bg=PANEL, fg=MUTED, font=("Courier", 8)).pack(anchor="w")
        row_e = tk.Frame(row_inp, bg=PANEL)
        row_e.pack(fill="x")
        self._entry_input = tk.Entry(row_e, textvariable=self.var_input,
                                     bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                                     relief="flat", font=("Courier", 10), bd=0,
                                     highlightthickness=1,
                                     highlightbackground=BORDER,
                                     highlightcolor=ACCENT)
        self._entry_input.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 6))
        FlatButton(row_e, "…", command=self._escolher_input, small=True).pack(side="left")

        # Preview do input (clicável para lightbox)
        self._frame_input_preview = tk.Frame(sec_busca, bg=PANEL)
        self._frame_input_preview.pack(fill="x", padx=18, pady=(4, 0))
        self._lbl_input_preview = tk.Label(self._frame_input_preview, bg=PANEL,
                                           cursor="hand2", text="")
        self._lbl_input_preview.pack(side="left")
        self._lbl_input_preview.bind("<Button-1>", self._lightbox_input)

        row_top = tk.Frame(sec_busca, bg=PANEL)
        row_top.pack(fill="x", padx=18, pady=(8, 4))
        tk.Label(row_top, text="Resultados a exibir", bg=PANEL, fg=MUTED,
                 font=("Courier", 8)).pack(anchor="w")
        tk.Scale(row_top, variable=self.var_top, from_=3, to=30,
                 orient="horizontal", bg=PANEL, fg=TEXT,
                 troughcolor=SURFACE, activebackground=ACCENT,
                 highlightthickness=0, sliderrelief="flat",
                 bd=0, font=("Courier", 9)).pack(fill="x")

        FlatButton(sec_busca, "⌕  Buscar Similares",
                   command=self._iniciar_busca, accent=True).pack(
                       fill="x", padx=18, pady=(6, 12))

        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", padx=18, pady=2)

        # ── Seção de índice (carregar) ───────────────────────────────────────
        sec_indice = tk.Frame(outer, bg=PANEL)
        sec_indice.pack(fill="x", padx=0, pady=0)

        tk.Label(sec_indice, text="ÍNDICE ATUAL", bg=PANEL, fg=ACCENT,
                 font=("Courier", 9, "bold")).pack(anchor="w", padx=18, pady=(12, 4))
        self.lbl_indice_info = tk.Label(sec_indice,
                                        text="Nenhum índice carregado.",
                                        bg=PANEL, fg=MUTED,
                                        font=("Courier", 9),
                                        justify="left", wraplength=290)
        self.lbl_indice_info.pack(anchor="w", padx=18, pady=(0, 6))
        FlatButton(sec_indice, "Carregar Índice Existente",
                   command=self._carregar_indice, small=True).pack(
                       anchor="w", padx=18, pady=(0, 10))

        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", padx=18, pady=2)

        # ── Seção de indexação (recolhível) ──────────────────────────────────
        col = CollapseSection(outer, "CONFIGURAÇÕES — Criar / Atualizar Índice", bg=PANEL)
        col.pack(fill="x")

        body = col.body

        row_p = tk.Frame(body, bg=PANEL)
        row_p.pack(fill="x", padx=18, pady=(8, 4))
        tk.Label(row_p, text="Pasta raiz dos PDFs", bg=PANEL, fg=MUTED,
                 font=("Courier", 8)).pack(anchor="w")
        row_ep = tk.Frame(row_p, bg=PANEL)
        row_ep.pack(fill="x")
        tk.Entry(row_ep, textvariable=self.var_pasta_pdfs,
                 bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Courier", 10), bd=0,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT).pack(side="left", fill="x", expand=True,
                                             ipady=6, padx=(0, 6))
        FlatButton(row_ep, "…", command=self._escolher_pasta_pdfs,
                   small=True).pack(side="left")

        row_i = tk.Frame(body, bg=PANEL)
        row_i.pack(fill="x", padx=18, pady=(0, 6))
        tk.Label(row_i, text="Salvar índice em", bg=PANEL, fg=MUTED,
                 font=("Courier", 8)).pack(anchor="w")
        row_ei = tk.Frame(row_i, bg=PANEL)
        row_ei.pack(fill="x")
        tk.Entry(row_ei, textvariable=self.var_pasta_indice,
                 bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Courier", 10), bd=0,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT).pack(side="left", fill="x", expand=True,
                                             ipady=6, padx=(0, 6))
        FlatButton(row_ei, "…", command=self._escolher_pasta_indice,
                   small=True).pack(side="left")

        # Barra de progresso
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("DS.Horizontal.TProgressbar",
                        troughcolor=SURFACE, background=ACCENT,
                        darkcolor=ACCENT, lightcolor=ACCENT,
                        bordercolor=PANEL, thickness=4)
        self.progressbar = ttk.Progressbar(body, variable=self.var_progresso,
                                           maximum=100, mode="determinate",
                                           style="DS.Horizontal.TProgressbar")
        self.progressbar.pack(fill="x", padx=18, pady=(0, 6))

        FlatButton(body, "▶  Iniciar Indexação",
                   command=self._iniciar_indexacao, accent=True).pack(
                       fill="x", padx=18, pady=(0, 14))

        return outer

    def _build_right_panel(self, parent):
        frame = tk.Frame(parent, bg=DARK)

        hdr = tk.Frame(frame, bg=DARK)
        hdr.pack(fill="x", padx=18, pady=(12, 6))
        self.lbl_resultados = tk.Label(hdr, text="Resultados",
                                       bg=DARK, fg=MUTED, font=("Courier", 11))
        self.lbl_resultados.pack(side="left")

        # ScrollableResults ocupa todo o espaço restante
        self.results_area = ScrollableResults(frame)
        self.results_area.pack(fill="both", expand=True)

        tk.Label(self.results_area.inner,
                 text="Carregue um índice e escolha um arquivo para buscar.",
                 bg=DARK, fg=MUTED, font=("Courier", 11)).pack(pady=80)

        return frame

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=SURFACE, height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self.var_status, bg=SURFACE, fg=MUTED,
                 font=("Courier", 9), anchor="w").pack(side="left", padx=12)

    # ── File pickers ────────────────────────────────────────────────────────

    def _escolher_pasta_pdfs(self):
        p = filedialog.askdirectory(title="Selecione a pasta raiz com os PDFs")
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
            self._carregar_preview_input(p)

    def _carregar_preview_input(self, caminho):
        def _worker():
            try:
                img = carregar_imagem_input(caminho)
                self._input_pil = img
                thumb = img.copy()
                thumb.thumbnail((80, 80), Image.LANCZOS)
                tk_img = ImageTk.PhotoImage(thumb)
                self.after(0, lambda: (
                    self._lbl_input_preview.config(image=tk_img),
                    setattr(self._lbl_input_preview, "_img", tk_img)
                ))
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()

    # ── Lightbox ────────────────────────────────────────────────────────────

    def _lightbox_input(self, _=None):
        if self._input_pil:
            Lightbox(self, self._input_pil,
                     titulo=Path(self.var_input.get()).name)

    def _lightbox_resultado(self, caminho):
        if caminho in self._thumb_refs:
            Lightbox(self, self._thumb_refs[caminho],
                     titulo=Path(caminho).name)

    # ── Modelo ──────────────────────────────────────────────────────────────

    def _carregar_modelo_background(self):
        def _worker():
            try:
                model, preprocess, device = carregar_modelo()
                with self._model_lock:
                    self.model      = model
                    self.preprocess = preprocess
                    self.device     = device
                label = "✓ Modelo de IA pronto"
                self.after(0, lambda: (
                    self.lbl_model_status.config(text=label, fg=SUCCESS),
                    self.var_status.set("Pronto.")
                ))
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
        self.after(0, lambda: (self.var_status.set("Iniciando indexação…"),
                               self.var_progresso.set(0)))
        Path(pasta_saida).mkdir(parents=True, exist_ok=True)

        pdfs = sorted(set(
            list(Path(pasta_pdfs).rglob("*.pdf")) +
            list(Path(pasta_pdfs).rglob("*.PDF"))
        ))

        if not pdfs:
            self.after(0, lambda: messagebox.showwarning(
                "Aviso", f"Nenhum PDF encontrado em:\n{pasta_pdfs}"))
            return

        total = len(pdfs)
        todos_emb, todos_meta = [], []

        for i, pdf_path in enumerate(pdfs):
            pct = int((i / total) * 100)
            nome = pdf_path.name
            self.after(0, lambda p=pct, n=nome: (
                self.var_progresso.set(p),
                self.var_status.set(f"[{p}%]  {n}")
            ))
            paginas = pdf_para_imagens(pdf_path)
            if not paginas:
                continue
            for inicio in range(0, len(paginas), BATCH_SIZE):
                lote     = paginas[inicio:inicio + BATCH_SIZE]
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
            self.after(0, lambda: messagebox.showerror("Erro", "Nenhum embedding gerado."))
            return

        matriz = np.stack(todos_emb)
        np.save(str(Path(pasta_saida) / "embeddings.npy"), matriz)
        with open(Path(pasta_saida) / "metadados.json", "w", encoding="utf-8") as f:
            json.dump(todos_meta, f, ensure_ascii=False, indent=2)

        self.embeddings = matriz
        self.metadados  = todos_meta
        n = len(todos_meta)
        self.after(0, lambda: (
            self.var_progresso.set(100),
            self.var_status.set(f"✓ Indexação concluída — {n} páginas."),
            self.lbl_indice_info.config(
                text=f"{n} páginas  |  {total} PDFs\n{pasta_saida}", fg=SUCCESS),
            messagebox.showinfo("Concluído",
                                f"Indexação concluída!\n{n} páginas de {total} PDFs.")
        ))

    # ── Carregar índice existente ────────────────────────────────────────────

    def _carregar_indice(self):
        pasta = filedialog.askdirectory(title="Selecione a pasta do índice")
        if not pasta:
            return
        emb_p  = Path(pasta) / "embeddings.npy"
        meta_p = Path(pasta) / "metadados.json"
        if not emb_p.exists() or not meta_p.exists():
            messagebox.showerror("Erro",
                                 "Pasta inválida: não contém embeddings.npy e metadados.json")
            return
        try:
            self.embeddings = np.load(str(emb_p))
            with open(meta_p, "r", encoding="utf-8") as f:
                self.metadados = json.load(f)
            n = len(self.metadados)
            self.lbl_indice_info.config(
                text=f"{n} páginas carregadas\n{pasta}", fg=SUCCESS)
            self.var_status.set(f"Índice carregado: {n} páginas.")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao carregar índice:\n{e}")

    # ── Busca ────────────────────────────────────────────────────────────────

    def _iniciar_busca(self):
        if self.embeddings is None or self.metadados is None:
            messagebox.showwarning("Aviso",
                                   "Nenhum índice carregado.\n"
                                   "Carregue um índice ou indexe um acervo primeiro.")
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

            scores  = (self.embeddings @ query.T).flatten()
            ranking = np.argsort(scores)[::-1]

            vistos = {}
            for idx in ranking:
                arq = self.metadados[idx]["arquivo"]
                if arq not in vistos:
                    vistos[arq] = (idx, float(scores[idx]))
                if len(vistos) >= self.var_top.get():
                    break

            resultados = [
                {
                    "rank":             i + 1,
                    "score":            score,
                    "arquivo":          arq,
                    "pagina":           self.metadados[idx]["pagina"] + 1,
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

    # ── Exibição de resultados ───────────────────────────────────────────────

    def _exibir_resultados(self, resultados):
        # Limpa tudo
        for w in self.results_area.inner.winfo_children():
            w.destroy()
        self._thumb_refs.clear()
        self._thumb_tk.clear()

        n = len(resultados)
        self.lbl_resultados.config(
            text=f"{n} resultado{'s' if n != 1 else ''} encontrado{'s' if n != 1 else ''}",
            fg=TEXT)
        self.var_status.set(f"Busca concluída — {n} resultado(s).")
        self.results_area.scroll_to_top()

        COLS = 4
        grid = tk.Frame(self.results_area.inner, bg=DARK)
        grid.pack(fill="both", expand=True, padx=12, pady=12)
        for col in range(COLS):
            grid.columnconfigure(col, weight=1)

        for r in resultados:
            col = (r["rank"] - 1) % COLS
            row = (r["rank"] - 1) // COLS
            self._criar_card(grid, r, row, col)

    def _criar_card(self, parent, resultado, row, col):
        caminho = resultado["caminho_completo"]

        card = tk.Frame(parent, bg=SURFACE, bd=0,
                        highlightthickness=1, highlightbackground=BORDER)
        card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

        # Thumbnail clicável
        thumb_lbl = tk.Label(card, bg=SURFACE, cursor="hand2",
                             width=PREVIEW_W, height=PREVIEW_H)
        thumb_lbl.pack(fill="x")
        thumb_lbl.bind("<Button-1>",
                       lambda _, c=caminho: self._lightbox_resultado(c))
        self._carregar_thumbnail_async(caminho, thumb_lbl)

        # Barra de score
        score   = resultado["score"]
        bar_frm = tk.Frame(card, bg=SURFACE)
        bar_frm.pack(fill="x", padx=8, pady=(4, 0))
        bar_w  = PREVIEW_W - 16
        filled = max(4, int(score * bar_w))
        tk.Frame(bar_frm, bg=ACCENT,  width=filled,          height=3).pack(side="left")
        tk.Frame(bar_frm, bg=BORDER,  width=bar_w - filled,  height=3).pack(side="left")

        # Info
        info = tk.Frame(card, bg=SURFACE)
        info.pack(fill="x", padx=8, pady=(4, 2))

        nome = Path(resultado["arquivo"]).name
        if len(nome) > 26:
            nome = nome[:23] + "…"
        tk.Label(info, text=nome, bg=SURFACE, fg=TEXT,
                 font=("Courier", 8, "bold"),
                 wraplength=PREVIEW_W - 16, justify="left",
                 anchor="w").pack(anchor="w")

        subpasta = str(Path(resultado["arquivo"]).parent)
        if subpasta != ".":
            sp = subpasta if len(subpasta) <= 28 else "…" + subpasta[-25:]
            tk.Label(info, text=sp, bg=SURFACE, fg=MUTED,
                     font=("Courier", 7),
                     wraplength=PREVIEW_W - 16, justify="left").pack(anchor="w")

        tk.Label(info, text=f"similaridade: {score:.1%}", bg=SURFACE, fg=ACCENT,
                 font=("Courier", 8)).pack(anchor="w")

        # Botão abrir no Explorer
        btn_frm = tk.Frame(card, bg=SURFACE)
        btn_frm.pack(fill="x", padx=8, pady=(2, 8))
        FlatButton(btn_frm, "📂  Abrir no Explorer",
                   command=lambda c=caminho: abrir_pasta_do_arquivo(c),
                   small=True).pack(fill="x")

    def _carregar_thumbnail_async(self, caminho_pdf, label):
        if caminho_pdf in self._thumb_tk:
            label.config(image=self._thumb_tk[caminho_pdf])
            return

        def _worker():
            try:
                paginas = pdf_para_imagens(Path(caminho_pdf), max_paginas=1, dpi=120)
                if not paginas:
                    raise ValueError("sem páginas")
                full_img = paginas[0][1]
                thumb    = gerar_thumbnail(full_img, PREVIEW_W, PREVIEW_H)
                tk_img   = ImageTk.PhotoImage(thumb)
                # Guarda full_img para lightbox
                self._thumb_refs[caminho_pdf] = full_img
                self._thumb_tk[caminho_pdf]   = tk_img
                self.after(0, lambda lbl=label, im=tk_img:
                           lbl.config(image=im))
            except Exception:
                placeholder = Image.new("RGB", (PREVIEW_W, PREVIEW_H), (40, 40, 50))
                tk_img = ImageTk.PhotoImage(placeholder)
                self._thumb_tk[caminho_pdf] = tk_img
                self.after(0, lambda lbl=label, im=tk_img: lbl.config(image=im))

        threading.Thread(target=_worker, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()