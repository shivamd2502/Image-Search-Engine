"""
finetuned_clip_search.py  — v2
───────────────────────────────
Uses your PEFT-finetuned CLIPWithHeads model for BOTH embedding cache
construction AND query encoding, so they are always in the same space.

The embedding cache is fully self-contained (no EmbeddingManager dependency
for encoding), stored as embeddings_cache_finetuned.pkl in the image folder.

WHERE TO PUT best_model.pt
───────────────────────────
Same folder as this script:
  Image-Search-by-Text-Prompt-using-CLIP/
  ├── best_model.pt
  ├── finetuned_clip_search.py   ← this file
  ├── embedding_manager.py       (not used for encoding anymore)
  └── ...
"""

import os
import time
import pickle
import random
import threading

import torch
import clip                     # pip install git+https://github.com/openai/CLIP.git
import numpy as np
from PIL import Image, ImageOps
from torch import nn
import torch.nn.functional as F
from tkinter import filedialog
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox

# ── Config ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(SCRIPT_DIR, "best_model.pt")
CACHE_FILE   = "embeddings_cache_finetuned.pkl"   # saved inside the image folder
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

DEFAULT_CLASSES = ["cloudy", "desert", "green_area", "water"]

CLASS_PROMPTS = {
    "cloudy":     ["a satellite image of a cloudy region",
                   "an aerial view covered with clouds",
                   "a satellite photo showing heavy cloud cover"],
    "desert":     ["a satellite image of a desert region",
                   "an aerial view of sandy desert terrain",
                   "a top-down satellite photo of dry desert land"],
    "green_area": ["a satellite image of green vegetation",
                   "an aerial view of dense green forest",
                   "a satellite photo showing lush green land"],
    "water":      ["a satellite image of a water body",
                   "an aerial view of a river or lake",
                   "a satellite photo of an ocean region"],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Model definition  (mirrors Kaggle Cell 6 exactly)
# ══════════════════════════════════════════════════════════════════════════════
class CLIPWithHeads(nn.Module):
    def __init__(self, classes, num_classes, device="cpu"):
        super().__init__()
        self.classes = classes
        self.device  = device

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.float()

        embed_dim = self.clip_model.visual.output_dim  # 512

        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.image_head  = nn.Linear(embed_dim, num_classes)
        self.text_head   = nn.Linear(embed_dim, num_classes)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    # ── encoders ──────────────────────────────────────────────────────────
    def encode_image(self, images):
        """Frozen CLIP → adapter → L2-norm  (finetuned image embedding)."""
        with torch.no_grad():
            x = self.clip_model.encode_image(images)
        x = x.float()
        x = self.adapter(x)
        x = F.normalize(x, dim=-1)
        return x

    def encode_text_tokens(self, tokens):
        """Frozen CLIP text encoder → L2-norm  (NO adapter, matches training)."""
        with torch.no_grad():
            x = self.clip_model.encode_text(tokens)
        x = x.float()
        x = F.normalize(x, dim=-1)
        return x

    def classify_image(self, images):
        feat   = self.encode_image(images)
        logits = self.image_head(feat)
        return logits.argmax(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  Load checkpoint
# ══════════════════════════════════════════════════════════════════════════════
def load_model(ckpt_path: str, device: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{ckpt_path}\n\n"
            "Download best_model.pt from your Kaggle Output tab and place it "
            "in the same folder as this script."
        )
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes    = ckpt.get("classes", DEFAULT_CLASSES)
    num_cls    = len(classes)
    model      = CLIPWithHeads(classes=classes, num_classes=num_cls, device=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[model] loaded — epoch {ckpt['epoch']+1}, "
          f"val_acc {ckpt['val_acc']:.2f}%, classes {classes}")
    return model, model.preprocess, classes


# ══════════════════════════════════════════════════════════════════════════════
#  Self-contained embedding cache  (no EmbeddingManager)
# ══════════════════════════════════════════════════════════════════════════════
def _list_images(folder: str):
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(root, f))
    return paths


def _encode_image_file(path: str, model: CLIPWithHeads,
                        preprocess, device: str) -> np.ndarray:
    img   = Image.open(path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img_t)
    return feat.cpu().numpy().flatten()          # shape (512,)


def build_cache(folder: str, model: CLIPWithHeads, preprocess,
                device: str, progress_cb=None) -> dict:
    """
    Build / update the finetuned embedding cache.
    Returns dict: {abs_image_path: np.ndarray(512,)}
    progress_cb(done, total) called after each image.
    """
    cache_path = os.path.join(folder, CACHE_FILE)

    # Load existing cache
    cache: dict = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        # Normalise all stored embeddings to flat (512,) — guards against
        # old caches that stored (1,512) or other shapes
        cache = {k: np.array(v).flatten() for k, v in cache.items()}
        print(f"[cache] loaded {len(cache)} existing embeddings")

    images      = _list_images(folder)
    current_set = set(images)
    cached_set  = set(cache.keys())

    # Remove stale entries (deleted images)
    stale = cached_set - current_set
    for p in stale:
        del cache[p]
    if stale:
        print(f"[cache] removed {len(stale)} stale entries")

    # Encode new images
    new_images = [p for p in images if p not in cache]
    print(f"[cache] {len(new_images)} new images to encode")

    for i, path in enumerate(new_images):
        try:
            cache[path] = _encode_image_file(path, model, preprocess, device)
        except Exception as e:
            print(f"[cache] skip {path}: {e}")
        if progress_cb:
            progress_cb(i + 1, len(new_images))

    # Save
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"[cache] saved {len(cache)} embeddings → {cache_path}")
    return cache


def encode_query(query: str, model: CLIPWithHeads,
                 device: str, classes: list) -> np.ndarray:
    """
    Encode a free-form text query.
    Averages the raw query with any matching class prompts.
    Returns np.ndarray (512,).
    """
    sentences = [query]
    q_lower   = query.lower()
    for cls, prompts in CLASS_PROMPTS.items():
        display = cls.replace("_", " ")
        if display in q_lower or cls in q_lower:
            sentences.extend(prompts)

    tokens = clip.tokenize(sentences, truncate=True).to(device)
    embs   = model.encode_text_tokens(tokens)          # [N, 512]
    avg    = embs.mean(dim=0)
    avg    = F.normalize(avg, dim=0)
    return avg.cpu().numpy()                           # (512,)


def search(query: str, cache: dict, model: CLIPWithHeads,
           device: str, classes: list,
           top_k: int = 8, threshold: float = 0.0):
    """Return [(path, score), ...] sorted by descending similarity."""
    text_feat = encode_query(query, model, device, classes)   # (512,)

    results = []
    for img_path, img_emb in cache.items():
        sim = float(np.dot(text_feat, img_emb))
        if sim >= threshold:
            results.append((img_path, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════════════════
class App(ctk.CTk):
    COLORS = {
        "primary":   "#1f538d",
        "secondary": "#14375e",
        "accent":    "#ffc107",
        "success":   "#28a745",
        "warning":   "#fd7e14",
        "danger":    "#dc3545",
    }

    def __init__(self):
        super().__init__()
        self.title("🛰️ Finetuned CLIP — Land Cover Search")
        self.geometry("1400x900")
        self.minsize(1000, 700)

        self.device     = ("mps"  if torch.backends.mps.is_available() else
                           "cuda" if torch.cuda.is_available()          else "cpu")
        self.model      = None
        self.preprocess = None
        self.classes    = DEFAULT_CLASSES
        self.cache      : dict = {}
        self.folder     : str  = ""
        self.results    : list = []
        self.loading    = True
        self.running    = False

        self._build_ui()
        threading.Thread(target=self._load_model_bg, daemon=True).start()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        root = ctk.CTkFrame(self, corner_radius=0)
        root.grid(sticky="nsew")
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)

        self._hdr(root)

        body = ctk.CTkFrame(root, corner_radius=0)
        body.grid(row=1, sticky="nsew")
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self._sidebar(body)
        self._results_panel(body)
        self._statusbar(root)

    # header
    def _hdr(self, p):
        h = ctk.CTkFrame(p, height=75, corner_radius=0)
        h.grid(row=0, sticky="ew"); h.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(h, text="🛰️ Finetuned CLIP — Land Cover Image Search",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#ffc107").grid(row=0, column=0, padx=20, pady=18, sticky="w")
        ctk.CTkLabel(h, text="PEFT adapter · ViT-B/32 · 99.88 % test accuracy",
                     font=ctk.CTkFont(size=12), text_color="gray"
                     ).grid(row=0, column=1, padx=20, sticky="e")

    # sidebar
    def _sidebar(self, p):
        sb = ctk.CTkFrame(p, width=340, corner_radius=10)
        sb.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)
        sb.grid_propagate(False)

        # folder
        fc = ctk.CTkFrame(sb, corner_radius=10)
        fc.pack(padx=15, pady=(15,5), fill="x")
        ctk.CTkLabel(fc, text="📁 Image Folder",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(12,6))
        self.folder_btn = ctk.CTkButton(
            fc, text="Select Folder", height=38, corner_radius=19,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._pick_folder)
        self.folder_btn.pack(padx=15, fill="x")
        self.folder_lbl = ctk.CTkLabel(
            fc, text="No folder selected", font=ctk.CTkFont(size=10),
            text_color="gray", wraplength=290)
        self.folder_lbl.pack(padx=15, pady=(6,4))
        self.cache_lbl = ctk.CTkLabel(
            fc, text="", font=ctk.CTkFont(size=10), text_color="gray")
        self.cache_lbl.pack(padx=15, pady=(0,12))

        # query
        qc = ctk.CTkFrame(sb, corner_radius=10)
        qc.pack(padx=15, pady=5, fill="x")
        ctk.CTkLabel(qc, text="🔎 Search Query",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(12,6))
        self.entry = ctk.CTkEntry(
            qc, placeholder_text="e.g. green area, water body, desert …",
            height=42, corner_radius=21, font=ctk.CTkFont(size=13), border_width=2)
        self.entry.pack(padx=15, fill="x")
        self.entry.bind("<Return>", lambda e: self._do_search())

        # quick class buttons
        bf = ctk.CTkFrame(qc, fg_color="transparent")
        bf.pack(padx=15, pady=(6,0), fill="x")
        for cls in DEFAULT_CLASSES:
            ctk.CTkButton(
                bf, text=cls.replace("_"," "), height=26, corner_radius=13,
                font=ctk.CTkFont(size=11), fg_color="transparent",
                border_width=1, text_color=("gray20","gray80"),
                hover_color=("gray85","gray25"),
                command=lambda c=cls: self._set_query(c.replace("_"," "))
            ).pack(side="left", padx=2, pady=4)

        self.search_btn = ctk.CTkButton(
            qc, text="🚀 Search", height=42, corner_radius=21,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.COLORS["primary"],
            hover_color=self.COLORS["secondary"],
            command=self._do_search)
        self.search_btn.pack(padx=15, pady=(8,15), fill="x")

        # settings
        sc = ctk.CTkFrame(sb, corner_radius=10)
        sc.pack(padx=15, pady=5, fill="x")
        ctk.CTkLabel(sc, text="⚙️ Settings",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(12,8))

        ctk.CTkLabel(sc, text="Max Results", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=15)
        self.sl_results = ctk.CTkSlider(sc, from_=1, to=20, number_of_steps=19,
            command=lambda v: self.lbl_results.configure(text=f"{int(v)}"))
        self.sl_results.set(8); self.sl_results.pack(padx=15, fill="x")
        self.lbl_results = ctk.CTkLabel(sc, text="8", font=ctk.CTkFont(size=11), text_color="gray")
        self.lbl_results.pack(anchor="w", padx=15)

        ctk.CTkLabel(sc, text="Similarity Threshold",
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=15, pady=(8,0))
        self.sl_thresh = ctk.CTkSlider(sc, from_=0.0, to=0.9, number_of_steps=9,
            command=lambda v: self.lbl_thresh.configure(text=f"{v:.2f}"))
        self.sl_thresh.set(0.0); self.sl_thresh.pack(padx=15, fill="x")
        self.lbl_thresh = ctk.CTkLabel(sc, text="0.00",
                                        font=ctk.CTkFont(size=11), text_color="gray")
        self.lbl_thresh.pack(anchor="w", padx=15, pady=(0,4))

        ctk.CTkButton(
            sc, text="🔄 Rebuild Cache", height=32,
            font=ctk.CTkFont(size=11),
            fg_color=self.COLORS["secondary"], hover_color=self.COLORS["primary"],
            command=self._rebuild_cache
        ).pack(padx=15, pady=(4,14), fill="x")

        # progress bar (hidden until search)
        self.prog = ctk.CTkProgressBar(sb, mode="indeterminate", height=8)

        # action buttons
        ac = ctk.CTkFrame(sb, fg_color="transparent")
        ac.pack(padx=15, pady=5, fill="x")
        ctk.CTkButton(ac, text="🧹 Clear", height=32, font=ctk.CTkFont(size=12),
                      fg_color=self.COLORS["warning"], hover_color="#e68900",
                      command=self._clear).pack(pady=2, fill="x")
        ctk.CTkButton(ac, text="📤 Export", height=32, font=ctk.CTkFont(size=12),
                      fg_color=self.COLORS["success"], hover_color="#1e7e34",
                      command=self._export).pack(pady=2, fill="x")

    # results panel
    def _results_panel(self, p):
        rc = ctk.CTkFrame(p, corner_radius=10)
        rc.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=10)
        rc.grid_columnconfigure(0, weight=1)
        rc.grid_rowconfigure(1, weight=1)

        hdr = ctk.CTkFrame(rc, height=56, corner_radius=8)
        hdr.grid(row=0, sticky="ew", padx=10, pady=(10,5))
        hdr.grid_columnconfigure(1, weight=1)

        self.results_title = ctk.CTkLabel(
            hdr, text="🖼️ Results",
            font=ctk.CTkFont(size=18, weight="bold"))
        self.results_title.grid(row=0, column=0, padx=15, pady=14, sticky="w")

        self.view_seg = ctk.CTkSegmentedButton(
            hdr, values=["Grid","List","Detail"],
            command=self._rerender, font=ctk.CTkFont(size=12))
        self.view_seg.set("Grid")
        self.view_seg.grid(row=0, column=1, padx=15, sticky="e")

        self.scroll = ctk.CTkScrollableFrame(
            rc, corner_radius=8,
            scrollbar_button_color=self.COLORS["primary"],
            scrollbar_button_hover_color=self.COLORS["secondary"])
        self.scroll.grid(row=1, sticky="nsew", padx=10, pady=(0,10))

        self._welcome()

    # status bar
    def _statusbar(self, p):
        sb = ctk.CTkFrame(p, height=28, corner_radius=0)
        sb.grid(row=2, sticky="ew"); sb.grid_columnconfigure(1, weight=1)
        self.status_lbl = ctk.CTkLabel(sb, text="Loading model…",
                                        font=ctk.CTkFont(size=11), text_color="gray")
        self.status_lbl.grid(row=0, column=0, padx=10, sticky="w")
        self.model_lbl = ctk.CTkLabel(sb, text="🔄 Loading PEFT-CLIP…",
                                       font=ctk.CTkFont(size=11), text_color="orange")
        self.model_lbl.grid(row=0, column=1, padx=10, sticky="e")

    # ── Screens ───────────────────────────────────────────────────────────
    def _welcome(self):
        self._clear_scroll()
        f = ctk.CTkFrame(self.scroll, corner_radius=15)
        f.pack(pady=60, padx=60, fill="both", expand=True)
        ctk.CTkLabel(f, text="🛰️", font=ctk.CTkFont(size=72)).pack(pady=(40,10))
        ctk.CTkLabel(f, text="Finetuned CLIP Land Cover Search",
                     font=ctk.CTkFont(size=22, weight="bold")).pack()
        ctk.CTkLabel(f,
            text="1. Select a folder of satellite images\n"
                 "2. Wait for the cache to build\n"
                 "3. Type a description and hit Search",
            font=ctk.CTkFont(size=13), text_color="gray").pack(pady=(10,40))

    def _no_results(self):
        self._clear_scroll()
        f = ctk.CTkFrame(self.scroll, corner_radius=15)
        f.pack(pady=60, padx=60, fill="both", expand=True)
        ctk.CTkLabel(f, text="🔍", font=ctk.CTkFont(size=64)).pack(pady=(30,10))
        ctk.CTkLabel(f, text="No matching images found",
                     font=ctk.CTkFont(size=18, weight="bold")).pack()
        ctk.CTkLabel(f,
            text="Tips:\n"
                 "• Lower the Similarity Threshold to 0.00\n"
                 "• Try class names: cloudy, desert, green area, water\n"
                 "• Make sure the cache was built with this script (Rebuild Cache)",
            font=ctk.CTkFont(size=12), text_color="gray",
            justify="left").pack(pady=(8,30))

    # ── Model loading ─────────────────────────────────────────────────────
    def _load_model_bg(self):
        try:
            m, pre, cls = load_model(MODEL_PATH, self.device)
            self.model, self.preprocess, self.classes = m, pre, cls
            self.loading = False
            self.after(0, self._model_ok)
        except Exception as e:
            self.after(0, lambda: CTkMessagebox(
                title="Model Load Error", message=str(e), icon="cancel"))
            self.after(0, lambda: self.model_lbl.configure(
                text="❌ Model failed", text_color="red"))

    def _model_ok(self):
        self.model_lbl.configure(
            text=f"✅ PEFT-CLIP ready [{self.device.upper()}]",
            text_color=self.COLORS["success"])
        self.status_lbl.configure(text="Model ready. Select a folder to begin.")

    # ── Folder & cache ────────────────────────────────────────────────────
    def _pick_folder(self):
        path = filedialog.askdirectory(title="Select Image Folder")
        if not path:
            return
        if self.loading:
            CTkMessagebox(title="Not Ready",
                          message="Model is still loading, please wait.", icon="warning")
            return
        self.folder = path
        self.folder_lbl.configure(text=path)
        self.folder_btn.configure(text="📁 Change Folder")
        self._build_cache_bg(force=False)

    def _rebuild_cache(self):
        if not self.folder:
            CTkMessagebox(title="No Folder", message="Select a folder first.", icon="warning")
            return
        c = CTkMessagebox(title="Rebuild?",
                          message="Delete existing cache and recompute all embeddings?",
                          icon="question", option_1="Cancel", option_2="Rebuild")
        if c.get() == "Rebuild":
            cp = os.path.join(self.folder, CACHE_FILE)
            if os.path.exists(cp):
                os.remove(cp)
            self._build_cache_bg(force=True)

    def _build_cache_bg(self, force=False):
        self.cache_lbl.configure(text="⏳ Building cache…", text_color="orange")
        self.status_lbl.configure(text="Encoding images with finetuned model…")
        threading.Thread(target=self._cache_thread, daemon=True).start()

    def _cache_thread(self):
        try:
            def cb(done, total):
                self.after(0, lambda: self.cache_lbl.configure(
                    text=f"⏳ {done}/{total} encoded…", text_color="orange"))
                self.after(0, lambda: self.status_lbl.configure(
                    text=f"Encoding images: {done}/{total}"))

            self.cache = build_cache(
                self.folder, self.model, self.preprocess, self.device, progress_cb=cb)
            n = len(self.cache)
            self.after(0, lambda: self.cache_lbl.configure(
                text=f"✅ {n} images cached (finetuned features)",
                text_color=self.COLORS["success"]))
            self.after(0, lambda: self.status_lbl.configure(
                text=f"{n} images ready. Type a query and search!"))
        except Exception as e:
            msg = str(e)
            self.after(0, lambda: self.cache_lbl.configure(
                text="❌ Cache error", text_color=self.COLORS["danger"]))
            self.after(0, lambda: self.status_lbl.configure(text=msg))

    # ── Search ────────────────────────────────────────────────────────────
    def _do_search(self):
        if self.running or self.loading:
            return
        q = self.entry.get().strip()
        if not q:
            CTkMessagebox(title="Empty", message="Enter a search query.", icon="warning")
            return
        if not self.cache:
            CTkMessagebox(title="No Cache",
                          message="Select a folder first and wait for caching to finish.",
                          icon="warning")
            return

        self.running = True
        self.prog.pack(padx=15, pady=4, fill="x")
        self.prog.start()
        self.search_btn.configure(text="🔄 Searching…", state="disabled")
        self.status_lbl.configure(text=f"Searching for: {q}")

        threading.Thread(target=lambda: self._search_thread(q), daemon=True).start()

    def _search_thread(self, q):
        try:
            t0 = time.time()
            res = search(
                query     = q,
                cache     = self.cache,
                model     = self.model,
                device    = self.device,
                classes   = self.classes,
                top_k     = int(self.sl_results.get()),
                threshold = float(self.sl_thresh.get()),
            )
            elapsed = time.time() - t0
            self.results = res

            print(f"[search] '{q}' → {len(res)} results in {elapsed:.3f}s")
            if res:
                print(f"[search] score range: {res[-1][1]:.4f} – {res[0][1]:.4f}")
            self.after(0, lambda: self._show_results(res, elapsed, q))
        except Exception as e:
            self.after(0, lambda: CTkMessagebox(
                title="Search Error", message=str(e), icon="cancel"))
            self.after(0, lambda: self.status_lbl.configure(text=str(e)))
        finally:
            self.after(0, self._done_search)

    def _done_search(self):
        self.prog.stop(); self.prog.pack_forget()
        self.running = False
        self.search_btn.configure(text="🚀 Search", state="normal")

    # ── Results display ───────────────────────────────────────────────────
    def _show_results(self, res, elapsed, q):
        self._clear_scroll()
        if not res:
            self._no_results()
            return

        self.results_title.configure(
            text=f"🖼️ {len(res)} matches for \"{q}\"")
        self._rerender(self.view_seg.get())

        info = ctk.CTkFrame(self.scroll, corner_radius=10)
        info.pack(pady=16, padx=10, fill="x")
        ctk.CTkLabel(info,
            text=f"⚡ {elapsed:.3f}s  •  🎯 best {res[0][1]:.4f}  •  🤖 finetuned features",
            font=ctk.CTkFont(size=11), text_color="gray").pack(pady=8)

        self.status_lbl.configure(
            text=f"Done — {len(res)} result(s) in {elapsed:.3f}s")

    def _rerender(self, mode=None):
        if mode is None:
            mode = self.view_seg.get()
        # remove only image cards (not the stats bar)
        for w in list(self.scroll.winfo_children()):
            w.destroy()
        if not self.results:
            return
        if mode == "Grid":
            self._grid(self.results)
        elif mode == "List":
            self._list(self.results)
        else:
            self._detail(self.results)

    def _grid(self, res):
        cols = max(2, min(4, (self.winfo_width() - 400) // 300))
        for i, (p, s) in enumerate(res):
            self._card(p, s).grid(row=i//cols, column=i%cols,
                                   padx=10, pady=10, sticky="nsew")
            self.scroll.grid_columnconfigure(i % cols, weight=1)

    def _list(self, res):
        for p, s in res:
            self._card(p, s, size=(90, 90), horizontal=True).pack(
                pady=5, padx=10, fill="x")

    def _detail(self, res):
        for p, s in res:
            self._detail_card(p, s).pack(pady=10, padx=10, fill="x")

    # ── Card builders ─────────────────────────────────────────────────────
    def _card(self, path, score, size=(240,240), horizontal=False):
        card = ctk.CTkFrame(self.scroll, corner_radius=14)
        try:
            img    = Image.open(path)
            thumb  = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
            img_tk = ctk.CTkImage(thumb, size=size)

            if horizontal:
                card.grid_columnconfigure(1, weight=1)
                ctk.CTkLabel(card, image=img_tk, text="").grid(
                    row=0, column=0, padx=12, pady=12, sticky="w")
                inf = ctk.CTkFrame(card, fg_color="transparent")
                inf.grid(row=0, column=1, padx=12, pady=12, sticky="ew")
            else:
                ctk.CTkLabel(card, image=img_tk, text="").pack(padx=12, pady=(12,4))
                inf = ctk.CTkFrame(card, fg_color="transparent")
                inf.pack(padx=12, pady=(0,12), fill="x")

            ctk.CTkLabel(inf, text=os.path.basename(path),
                         font=ctk.CTkFont(size=11, weight="bold"),
                         wraplength=200).pack(pady=(0,3))
            ctk.CTkLabel(inf, text=f"🎯 {score:.4f}",
                         font=ctk.CTkFont(size=11),
                         text_color=self._score_color(score)).pack()

            bf = ctk.CTkFrame(inf, fg_color="transparent")
            bf.pack(pady=(5,0))
            ctk.CTkButton(bf, text="👁️", width=32, height=24,
                          font=ctk.CTkFont(size=10),
                          command=lambda: self._view(path)).pack(side="left", padx=2)
            ctk.CTkButton(bf, text="💾", width=32, height=24,
                          font=ctk.CTkFont(size=10),
                          command=lambda: self._save(path)).pack(side="left", padx=2)

            card.bind("<Enter>", lambda e, c=card: c.configure(
                border_width=2, border_color=self.COLORS["primary"]))
            card.bind("<Leave>", lambda e, c=card: c.configure(border_width=0))
        except Exception as e:
            ctk.CTkLabel(card, text=f"❌ {os.path.basename(path)}\n{e}",
                         font=ctk.CTkFont(size=10)).pack(pady=16)
        return card

    def _detail_card(self, path, score):
        card = ctk.CTkFrame(self.scroll, corner_radius=14)
        try:
            img   = Image.open(path)
            w, h  = img.size
            fsz   = os.path.getsize(path)
            thumb = ImageOps.fit(img, (140,140), Image.Resampling.LANCZOS)
            tk_im = ctk.CTkImage(thumb, size=(140,140))

            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(padx=18, pady=18, fill="x")
            row.grid_columnconfigure(1, weight=1)

            ctk.CTkLabel(row, image=tk_im, text="").grid(
                row=0, column=0, padx=(0,18), sticky="nw")

            col = ctk.CTkFrame(row, fg_color="transparent")
            col.grid(row=0, column=1, sticky="ew")

            ctk.CTkLabel(col, text=os.path.basename(path),
                         font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w")
            ctk.CTkLabel(col, text=f"🎯 Similarity: {score:.4f}",
                         font=ctk.CTkFont(size=13),
                         text_color=self._score_color(score)).pack(anchor="w", pady=2)
            ctk.CTkLabel(col, text=f"📐 {w}×{h}  💾 {self._fmt(fsz)}",
                         font=ctk.CTkFont(size=11), text_color="gray").pack(anchor="w")

            pred = self._predict(path)
            if pred:
                ctk.CTkLabel(col, text=f"🏷️ Predicted class: {pred}",
                             font=ctk.CTkFont(size=12),
                             text_color=self.COLORS["accent"]).pack(anchor="w", pady=2)

            bf = ctk.CTkFrame(col, fg_color="transparent")
            bf.pack(anchor="w", pady=(6,0))
            ctk.CTkButton(bf, text="👁️ View", width=80, height=28,
                          command=lambda: self._view(path)).pack(side="left", padx=(0,6))
            ctk.CTkButton(bf, text="💾 Save", width=80, height=28,
                          command=lambda: self._save(path)).pack(side="left")
        except Exception as e:
            ctk.CTkLabel(card, text=f"❌ {e}").pack(pady=16)
        return card

    # ── Helpers ───────────────────────────────────────────────────────────
    def _predict(self, path):
        try:
            img   = Image.open(path).convert("RGB")
            img_t = self.preprocess(img).unsqueeze(0).to(self.device)
            idx   = self.model.classify_image(img_t).item()
            return self.classes[idx].replace("_", " ")
        except Exception:
            return None

    def _score_color(self, s):
        if s > 0.6:  return self.COLORS["success"]
        if s > 0.4:  return self.COLORS["accent"]
        if s > 0.2:  return self.COLORS["warning"]
        return self.COLORS["danger"]

    def _fmt(self, b):
        if b < 1024:    return f"{b}B"
        if b < 1<<20:   return f"{b/1024:.1f}KB"
        return f"{b/(1<<20):.1f}MB"

    def _set_query(self, t):
        self.entry.delete(0, "end"); self.entry.insert(0, t)

    def _clear_scroll(self):
        for w in self.scroll.winfo_children():
            w.destroy()

    def _clear(self):
        self.results = []
        self.results_title.configure(text="🖼️ Results")
        self._welcome()

    def _view(self, path):
        pop = ctk.CTkToplevel(self)
        pop.title(os.path.basename(path))
        pop.geometry("820x680")
        try:
            img = Image.open(path)
            img.thumbnail((760, 560), Image.Resampling.LANCZOS)
            tk_im = ctk.CTkImage(img, size=img.size)
            ctk.CTkLabel(pop, image=tk_im, text="").pack(pady=(18,6), expand=True)
            pred = self._predict(path)
            if pred:
                ctk.CTkLabel(pop, text=f"🏷️ Finetuned model: {pred}",
                             font=ctk.CTkFont(size=14, weight="bold"),
                             text_color=self.COLORS["accent"]).pack(pady=(0,16))
        except Exception as e:
            ctk.CTkLabel(pop, text=str(e)).pack(pady=40)

    def _save(self, path):
        dst = filedialog.asksaveasfilename(
            initialfile=os.path.basename(path),
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("All","*.*")])
        if dst:
            try:
                Image.open(path).save(dst)
                CTkMessagebox(title="Saved", message="Image saved!", icon="check")
            except Exception as e:
                CTkMessagebox(title="Error", message=str(e), icon="cancel")

    def _export(self):
        if not self.results:
            CTkMessagebox(title="Empty", message="No results to export.", icon="warning")
            return
        dst = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text","*.txt"),("CSV","*.csv")])
        if dst:
            with open(dst, "w") as f:
                f.write("Finetuned CLIP — Search Results\n" + "="*50 + "\n\n")
                for i,(p,s) in enumerate(self.results,1):
                    pred = self._predict(p) or "?"
                    f.write(f"{i}. {os.path.basename(p)}\n"
                            f"   Score: {s:.6f}\n"
                            f"   Class: {pred}\n"
                            f"   Path:  {p}\n\n")
            CTkMessagebox(title="Done", message="Exported!", icon="check")


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()