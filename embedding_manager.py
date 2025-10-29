# embedding_manager.py
import os
import torch
import pickle
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import open_clip

class EmbeddingManager:
    def __init__(self, image_folder, model_name="ViT-B-32", cache_path="embeddings.pkl", device=None):
        self.image_folder = image_folder
        self.cache_path = cache_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and preprocessing
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained="laion2b_s34b_b79k"  # Fixed: added 'b' before 79k
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load cache if exists
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self.embeddings = pickle.load(f)
        else:
            self.embeddings = {}

    def compute_embedding(self, image_path):
        """Compute CLIP embedding for a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(image_tensor)
                emb /= emb.norm(dim=-1, keepdim=True)
            return emb.cpu().numpy()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def update_cache(self):
        """Compute embeddings for all missing or new images."""
        all_images = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
        new_images = [img for img in all_images if img not in self.embeddings]

        for img_path in new_images:
            emb = self.compute_embedding(img_path)
            if emb is not None:
                self.embeddings[img_path] = emb
                print(f"Added to cache: {os.path.basename(img_path)}")

        # Remove deleted images from cache
        cached_files = list(self.embeddings.keys())
        for img_path in cached_files:
            if not os.path.exists(img_path):
                del self.embeddings[img_path]
                print(f"Removed deleted image from cache: {os.path.basename(img_path)}")

        with open(self.cache_path, "wb") as f:
            pickle.dump(self.embeddings, f)
        print(f"Cache updated. Total embeddings stored: {len(self.embeddings)}")

    def get_embeddings(self):
        """Return all cached embeddings."""
        return self.embeddings

# Optional: Watchdog-based automation for real-time updates
class FolderWatcher(FileSystemEventHandler):
    def __init__(self, manager):
        self.manager = manager

    def on_created(self, event):
        if event.src_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            print(f"New image detected: {event.src_path}")
            self.manager.update_cache()

    def on_deleted(self, event):
        if event.src_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            print(f"Image deleted: {event.src_path}")
            self.manager.update_cache()

def start_watching(folder, manager):
    event_handler = FolderWatcher(manager)
    observer = Observer()
    observer.schedule(event_handler, folder, recursive=False)
    observer.start()
    print("Watching folder for new/deleted images...")
    return observer