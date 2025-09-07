import os
import io
import time
import torch
import open_clip
import threading
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox

# Configuration
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ImageSearchApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EDAI Image Finder with CLIP")
        self.geometry("1200x800")
        
        # Initialize AI Model
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.model.to(self.device)
        
        # UI Setup
        self.create_widgets()
        self.running = False
        
        # Start loading model in background
        self.loading = True
        threading.Thread(target=self.initialize_model, daemon=True).start()

    def create_widgets(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Control Panel
        left_frame = ctk.CTkFrame(self, width=300)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Search Controls
        ctk.CTkLabel(left_frame, text="Image Finder by Description", font=("Arial", 20, "bold")).pack(pady=20)
        
        self.folder_btn = ctk.CTkButton(left_frame, text="📁 Select Folder", command=self.select_folder)
        self.folder_btn.pack(pady=5, fill='x')
        
        self.folder_label = ctk.CTkLabel(left_frame, text="No folder selected", wraplength=250)
        self.folder_label.pack(pady=10)
        
        self.search_entry = ctk.CTkEntry(left_frame, placeholder_text="Enter search prompt...", height=40)
        self.search_entry.pack(pady=10, fill='x')
        self.search_entry.bind("<Return>", lambda e: self.start_search())
        
        self.search_btn = ctk.CTkButton(left_frame, text="🔍 Search", command=self.start_search)
        self.search_btn.pack(pady=5, fill='x')
        
        # Action Buttons
        self.clear_btn = ctk.CTkButton(left_frame, text="🧹 Clear Results", command=self.clear_results)
        self.clear_btn.pack(pady=5, fill='x')
        
        # Settings
        settings_frame = ctk.CTkFrame(left_frame)
        settings_frame.pack(pady=20, fill='x')
        
        ctk.CTkLabel(settings_frame, text="⚙️ Settings").pack(pady=5)
        self.results_slider = ctk.CTkSlider(settings_frame, from_=1, to=10, number_of_steps=9)
        self.results_slider.set(5)
        self.results_slider.pack(pady=10, padx=10, fill='x')
        ctk.CTkLabel(settings_frame, text="Number of Results").pack()
        
        # Progress Bar
        self.progress = ctk.CTkProgressBar(left_frame, mode='indeterminate')

        # Right Results Panel
        self.results_frame = ctk.CTkFrame(self)
        self.results_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_rowconfigure(0, weight=1)

        # Scrollable Canvas
        self.canvas = tk.Canvas(self.results_frame, bg='#2b2b2b', highlightthickness=0)
        self.scrollbar = ctk.CTkScrollbar(self.results_frame, orientation="vertical", command=self.canvas.yview)
        self.scrollable_frame = ctk.CTkFrame(self.canvas)
        
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def initialize_model(self):
        time.sleep(1)  # Simulate model loading
        self.loading = False
        self.after(0, lambda: self.folder_btn.configure(state="normal"))
        self.after(0, lambda: self.search_btn.configure(state="normal"))

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_label.configure(text=folder_path)
            self.animate_folder_select()

    def animate_folder_select(self):
        self.folder_label.configure(text_color="#4CAF50")
        self.after(1000, lambda: self.folder_label.configure(text_color=("black", "white")))

    def start_search(self):
        if self.running or self.loading:
            return
            
        prompt = self.search_entry.get().strip()
        folder = self.folder_label.cget("text")
        
        if not folder or folder == "No folder selected":
            CTkMessagebox(title="Error", message="Please select a folder first!", icon="cancel")
            return
            
        if not prompt:
            CTkMessagebox(title="Error", message="Please enter a search prompt!", icon="cancel")
            return
            
        self.running = True
        self.progress.pack(pady=10, fill='x')
        self.progress.start()
        
        threading.Thread(target=lambda: self.search_images(folder, prompt), daemon=True).start()

    def search_images(self, folder_path, prompt):
        try:
            start_time = time.time()
            images = load_images(folder_path)
            text_features = encode_text(prompt, self.model, self.device)
            
            results = []
            for img_path, img in images:
                img_features = encode_image(img, self.preprocess, self.model, self.device)
                similarity = (text_features @ img_features.T).item()
                results.append((img_path, similarity))
            
            results.sort(key=lambda x: x[1], reverse=True)
            final_results = results[:int(self.results_slider.get())]
            
            self.after(0, lambda: self.show_results(final_results, time.time() - start_time))
            
        except Exception as e:
            self.after(0, lambda: CTkMessagebox(title="Error", message=str(e), icon="cancel"))
        finally:
            self.after(0, self.reset_ui)

    def show_results(self, results, search_time):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        cols = 3
        for i, (img_path, score) in enumerate(results):
            row = i // cols
            col = i % cols
            
            frame = ctk.CTkFrame(self.scrollable_frame)
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            img = Image.open(img_path)
            img.thumbnail((300, 300))
            img_tk = ctk.CTkImage(img, size=img.size)
            label = ctk.CTkLabel(frame, image=img_tk, text="")
            label.pack(padx=5, pady=5)
            
            meta = f"📊 Score: {score:.2f}\n📂 {os.path.basename(img_path)}"
            ctk.CTkLabel(frame, text=meta).pack()
            
            # Action Buttons
            button_frame = ctk.CTkFrame(frame)
            button_frame.pack(pady=5)
            
            save_btn = ctk.CTkButton(
                button_frame, 
                text="💾 Save", 
                width=80,
                command=lambda p=img_path: self.save_image(p)
            )
            save_btn.pack(side="bottom", padx=5)
            
            # copy_btn = ctk.CTkButton(
            #     button_frame,
            #     text="📋 Copy",
            #     width=80,
            #     command=lambda p=img_path: self.copy_to_clipboard(p)
            # )
            # copy_btn.pack(side="left", padx=5)
            
            label.bind("<Enter>", lambda e, f=frame: f.configure(fg_color=("gray70", "gray30")))
            label.bind("<Leave>", lambda e, f=frame: f.configure(fg_color=("gray90", "gray20")))
            
        stats_frame = ctk.CTkFrame(self.scrollable_frame)
        stats_frame.grid(row=100, column=0, columnspan=3, pady=20, sticky="ew")
        ctk.CTkLabel(stats_frame, 
                    text=f"🔍 Found {len(results)} results in ⏱️ {search_time:.2f}s",
                    font=("Arial", 14)).pack(pady=10)
        
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def save_image(self, img_path):
        initial_file = os.path.basename(img_path)
        filetypes = (
            ("JPEG files", "*.jpg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        )
        save_path = filedialog.asksaveasfilename(
            title="Save Image As",
            initialfile=initial_file,
            defaultextension=".png",
            filetypes=filetypes
        )
        if save_path:
            try:
                img = Image.open(img_path)
                img.save(save_path)
                CTkMessagebox(title="Success", message="Image saved successfully!", icon="check")
            except Exception as e:
                CTkMessagebox(title="Error", message=f"Failed to save image: {str(e)}", icon="cancel")

    # def copy_to_clipboard(self, img_path):
    #     try:
    #         img = Image.open(img_path)
    #         output = io.BytesIO()
    #         img.save(output, format="PNG")
    #         data = output.getvalue()
    #         output.close()
            
    #         self.clipboard_clear()
    #         self.clipboard_append(data, type='image/png')
    #         CTkMessagebox(title="Success", message="Image copied to clipboard!", icon="check")
    #     except Exception as e:
    #         CTkMessagebox(title="Error", message=f"Failed to copy image: {str(e)}", icon="cancel")

    def clear_results(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        CTkMessagebox(title="Info", message="Results cleared successfully!", icon="info")

    def reset_ui(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.running = False

def load_images(folder_path):
    images = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            image_path = os.path.join(folder_path, file)
            try:
                images.append((image_path, Image.open(image_path).convert("RGB")))
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    return images

def encode_text(prompt, model, device):
    text_tokens = open_clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        return model.encode_text(text_tokens).cpu().numpy()

def encode_image(image, preprocess, model, device):
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(img_tensor).cpu().numpy()

if __name__ == "__main__":
    app = ImageSearchApp()
    app.mainloop()
