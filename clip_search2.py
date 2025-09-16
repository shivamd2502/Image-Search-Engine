import os
import io
import time
import torch
import open_clip
import threading
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import math

# Configuration
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ImageSearchApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🔍 EDAI Image Search Engine")
        self.geometry("1400x900")
        self.minsize(1000, 700)
        
        # Initialize AI Model
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.model.to(self.device)
        
        # App state
        self.running = False
        self.loading = True
        self.folder_path = None
        self.search_history = []
        self.current_results = []
        
        # Color scheme
        self.colors = {
            'primary': '#1f538d',
            'secondary': '#14375e',
            'accent': '#ffc107',
            'success': '#28a745',
            'warning': '#fd7e14',
            'danger': '#dc3545',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # UI Setup
        self.create_modern_ui()
        
        # Start loading model in background
        threading.Thread(target=self.initialize_model, daemon=True).start()
        
        # Bind window events
        self.bind("<Configure>", self.on_window_resize)
    
    def create_modern_ui(self):
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create main container
        main_container = ctk.CTkFrame(self, corner_radius=0)
        main_container.grid(row=0, column=0, columnspan=2, sticky="nsew")
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(1, weight=1)
        
        # Header with gradient effect
        self.create_header(main_container)
        
        # Main content area
        content_frame = ctk.CTkFrame(main_container, corner_radius=0)
        content_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=0)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Left sidebar
        self.create_sidebar(content_frame)
        
        # Right results area
        self.create_results_area(content_frame)
        
        # Status bar
        self.create_status_bar(main_container)

    def create_header(self, parent):
        header_frame = ctk.CTkFrame(parent, height=80, corner_radius=0)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
        header_frame.grid_columnconfigure(1, weight=1)
        
        # App title and icon
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        
        title_label = ctk.CTkLabel(
            title_frame, 
            text="🔍 EDAI Image Search Engine", 
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#ffc107"
        )
        title_label.pack(side="left")
        
        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="search image by text query",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        subtitle_label.pack(side="left", padx=(10, 0))
        
        # Header stats
        self.stats_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        self.stats_frame.grid(row=0, column=1, padx=20, pady=10, sticky="e")
        
        self.update_header_stats()

    def create_sidebar(self, parent):
        # Sidebar with improved styling
        sidebar = ctk.CTkFrame(parent, width=350, corner_radius=10)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        sidebar.grid_propagate(False)
        
        # Search section
        search_frame = ctk.CTkFrame(sidebar, corner_radius=10)
        search_frame.pack(pady=10, padx=15, fill="x")
        
        search_title = ctk.CTkLabel(
            search_frame, 
            text="🔎 Smart Search", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        search_title.pack(pady=(15, 10))
        
        # Folder selection with modern design
        folder_frame = ctk.CTkFrame(search_frame, fg_color="transparent")
        folder_frame.pack(pady=10, padx=15, fill="x")
        
        self.folder_btn = ctk.CTkButton(
            folder_frame,
            text="📁 Select Image Folder",
            command=self.select_folder,
            height=40,
            corner_radius=20,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.folder_btn.pack(fill="x")
        
        self.folder_display = ctk.CTkTextbox(
            folder_frame,
            height=60,
            corner_radius=10,
            font=ctk.CTkFont(size=11),
            wrap="word"
        )
        self.folder_display.pack(pady=(10, 0), fill="x")
        self.folder_display.insert("0.0", "No folder selected yet...\nClick above to choose your image directory")
        self.folder_display.configure(state="disabled")
        
        # Search input with suggestions
        search_input_frame = ctk.CTkFrame(search_frame, fg_color="transparent")
        search_input_frame.pack(pady=15, padx=15, fill="x")
        
        self.search_entry = ctk.CTkEntry(
            search_input_frame,
            placeholder_text="Describe what you're looking for...",
            height=45,
            corner_radius=22,
            font=ctk.CTkFont(size=14),
            border_width=2
        )
        self.search_entry.pack(fill="x")
        self.search_entry.bind("<Return>", lambda e: self.start_search())
        self.search_entry.bind("<KeyRelease>", self.on_search_input_change)
        
        # Search suggestions
        self.suggestions_frame = ctk.CTkFrame(search_input_frame, height=0)
        self.create_search_suggestions()
        
        # Search button with animation
        self.search_btn = ctk.CTkButton(
            search_input_frame,
            text="🚀 Search Images",
            command=self.start_search,
            height=45,
            corner_radius=22,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.colors['primary'],
            hover_color=self.colors['secondary']
        )
        self.search_btn.pack(pady=(10, 0), fill="x")
        
        # Advanced settings
        self.create_advanced_settings(sidebar)
        
        # Search history
        self.create_search_history(sidebar)
        
        # Action buttons
        self.create_action_buttons(sidebar)

    def create_search_suggestions(self):
        suggestions = [
            "a cat sleeping", "beautiful sunset", "modern architecture", 
            "people laughing", "colorful flowers", "vintage car",
            "mountain landscape", "abstract art", "food photography"
        ]
        
        suggestion_buttons = []
        for suggestion in suggestions[:3]:  # Show top 3
            btn = ctk.CTkButton(
                self.suggestions_frame,
                text=suggestion,
                height=25,
                font=ctk.CTkFont(size=11),
                command=lambda s=suggestion: self.use_suggestion(s),
                fg_color="transparent",
                text_color="gray",
                hover_color=("gray80", "gray20")
            )
            suggestion_buttons.append(btn)

    def create_advanced_settings(self, parent):
        settings_frame = ctk.CTkFrame(parent, corner_radius=10)
        settings_frame.pack(pady=10, padx=15, fill="x")
        
        settings_title = ctk.CTkLabel(
            settings_frame,
            text="⚙️ Search Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        settings_title.pack(pady=(15, 10))
        
        # Results count slider
        results_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        results_frame.pack(pady=10, padx=15, fill="x")
        
        ctk.CTkLabel(results_frame, text="Max Results:", font=ctk.CTkFont(size=12)).pack(anchor="w")
        
        self.results_slider = ctk.CTkSlider(
            results_frame,
            from_=1,
            to=20,
            number_of_steps=19,
            command=self.update_results_label
        )
        self.results_slider.set(8)
        self.results_slider.pack(pady=(5, 0), fill="x")
        
        self.results_label = ctk.CTkLabel(
            results_frame,
            text="8 results",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.results_label.pack(anchor="w")
        
        # Similarity threshold
        threshold_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        threshold_frame.pack(pady=10, padx=15, fill="x")
        
        ctk.CTkLabel(threshold_frame, text="Similarity Threshold:", font=ctk.CTkFont(size=12)).pack(anchor="w")
        
        self.threshold_slider = ctk.CTkSlider(
            threshold_frame,
            from_=0.1,
            to=1.0,
            number_of_steps=9,
            command=self.update_threshold_label
        )
        self.threshold_slider.set(0.2)
        self.threshold_slider.pack(pady=(5, 0), fill="x")
        
        self.threshold_label = ctk.CTkLabel(
            threshold_frame,
            text="0.2 threshold",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.threshold_label.pack(anchor="w")

    def create_search_history(self, parent):
        history_frame = ctk.CTkFrame(parent, corner_radius=10)
        history_frame.pack(pady=10, padx=15, fill="x")
        
        history_title = ctk.CTkLabel(
            history_frame,
            text="📜 Recent Searches",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        history_title.pack(pady=(15, 10))
        
        self.history_listbox = ctk.CTkTextbox(
            history_frame,
            height=100,
            corner_radius=8,
            font=ctk.CTkFont(size=11)
        )
        self.history_listbox.pack(pady=(0, 15), padx=15, fill="x")
        self.history_listbox.configure(state="disabled")

    def create_action_buttons(self, parent):
        action_frame = ctk.CTkFrame(parent, fg_color="transparent")
        action_frame.pack(pady=10, padx=15, fill="x")
        
        # Clear results button
        self.clear_btn = ctk.CTkButton(
            action_frame,
            text="🧹 Clear Results",
            command=self.clear_results,
            height=35,
            fg_color=self.colors['warning'],
            hover_color="#e68900",
            font=ctk.CTkFont(size=12)
        )
        self.clear_btn.pack(pady=2, fill="x")
        
        # Export results button
        self.export_btn = ctk.CTkButton(
            action_frame,
            text="📤 Export Results",
            command=self.export_results,
            height=35,
            fg_color=self.colors['success'],
            hover_color="#1e7e34",
            font=ctk.CTkFont(size=12)
        )
        self.export_btn.pack(pady=2, fill="x")
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(action_frame, mode='indeterminate', height=8)

    def create_results_area(self, parent):
        # Results container with modern styling
        results_container = ctk.CTkFrame(parent, corner_radius=10)
        results_container.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        results_container.grid_columnconfigure(0, weight=1)
        results_container.grid_rowconfigure(1, weight=1)
        
        # Results header
        results_header = ctk.CTkFrame(results_container, height=60, corner_radius=8)
        results_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        results_header.grid_columnconfigure(1, weight=1)
        
        # Results title and info
        self.results_title = ctk.CTkLabel(
            results_header,
            text="🖼️ Search Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.results_title.grid(row=0, column=0, padx=15, pady=15, sticky="w")
        
        # View options
        view_frame = ctk.CTkFrame(results_header, fg_color="transparent")
        view_frame.grid(row=0, column=1, padx=15, pady=15, sticky="e")
        
        self.view_mode = ctk.CTkSegmentedButton(
            view_frame,
            values=["Grid", "List", "Detailed"],
            command=self.change_view_mode,
            font=ctk.CTkFont(size=12)
        )
        self.view_mode.set("Grid")
        self.view_mode.pack(side="right")
        
        # Scrollable results frame
        self.results_scrollable = ctk.CTkScrollableFrame(
            results_container,
            corner_radius=8,
            scrollbar_button_color=self.colors['primary'],
            scrollbar_button_hover_color=self.colors['secondary']
        )
        self.results_scrollable.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        # Welcome message
        self.show_welcome_message()

    def create_status_bar(self, parent):
        status_frame = ctk.CTkFrame(parent, height=30, corner_radius=0)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready to search images...",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.model_status = ctk.CTkLabel(
            status_frame,
            text="🔄 Loading AI Model...",
            font=ctk.CTkFont(size=11),
            text_color="orange"
        )
        self.model_status.grid(row=0, column=1, padx=10, pady=5, sticky="e")

    def show_welcome_message(self):
        welcome_frame = ctk.CTkFrame(self.results_scrollable, corner_radius=15)
        welcome_frame.pack(pady=50, padx=50, fill="both", expand=True)
        
        # Large icon
        icon_label = ctk.CTkLabel(
            welcome_frame,
            text="🔍",
            font=ctk.CTkFont(size=80)
        )
        icon_label.pack(pady=(40, 20))
        
        # Welcome text
        welcome_title = ctk.CTkLabel(
            welcome_frame,
            text="Welcome to Image Search Engine",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        welcome_title.pack(pady=10)
        
        welcome_subtitle = ctk.CTkLabel(
            welcome_frame,
            text="Select a folder and describe what you're looking for",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        welcome_subtitle.pack(pady=(0, 40))

    def update_results_label(self, value):
        self.results_label.configure(text=f"{int(value)} results")

    def update_threshold_label(self, value):
        self.threshold_label.configure(text=f"{value:.1f} threshold")

    def update_header_stats(self):
        # Clear existing stats
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        if self.folder_path:
            # Count images in folder
            image_count = len([f for f in os.listdir(self.folder_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
            
            stats_label = ctk.CTkLabel(
                self.stats_frame,
                text=f"📁 {image_count} images • 🔍 {len(self.search_history)} searches",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            )
            stats_label.pack()

    def on_search_input_change(self, event):
        # Auto-complete or suggestions could be implemented here
        pass

    def use_suggestion(self, suggestion):
        self.search_entry.delete(0, "end")
        self.search_entry.insert(0, suggestion)

    def change_view_mode(self, value):
        if self.current_results:
            self.display_results_in_mode(self.current_results, value.lower())

    def on_window_resize(self, event):
        if event.widget == self:
            # Adjust grid layout based on window size
            if self.winfo_width() < 1200:
                self.grid_columnconfigure(0, weight=1, minsize=300)
            else:
                self.grid_columnconfigure(0, weight=0, minsize=350)

    def initialize_model(self):
        time.sleep(2)  # Simulate model loading
        self.loading = False
        self.after(0, self.model_loaded)

    def model_loaded(self):
        self.model_status.configure(
            text="✅ AI Model Ready",
            text_color=self.colors['success']
        )
        self.status_label.configure(text="AI model loaded successfully. Ready to search!")

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        if folder_path:
            self.folder_path = folder_path
            
            # Update UI
            self.folder_display.configure(state="normal")
            self.folder_display.delete("0.0", "end")
            self.folder_display.insert("0.0", f"📁 Selected Folder:\n{folder_path}")
            self.folder_display.configure(state="disabled")
            
            # Animate button
            original_color = self.folder_btn.cget("fg_color")
            self.folder_btn.configure(
                fg_color=self.colors['success'],
                text="✅ Folder Selected"
            )
            
            self.after(2000, lambda: self.folder_btn.configure(
                fg_color=original_color,
                text="📁 Change Folder"
            ))
            
            self.update_header_stats()
            self.status_label.configure(text=f"Folder selected: {os.path.basename(folder_path)}")

    def start_search(self):
        if self.running or self.loading:
            return
            
        prompt = self.search_entry.get().strip()
        
        if not self.folder_path:
            CTkMessagebox(
                title="No Folder Selected",
                message="Please select an image folder first!",
                icon="warning"
            )
            return
            
        if not prompt:
            CTkMessagebox(
                title="Empty Search",
                message="Please enter a search description!",
                icon="warning"
            )
            return
            
        # Add to search history
        if prompt not in self.search_history:
            self.search_history.append(prompt)
            if len(self.search_history) > 10:
                self.search_history.pop(0)
            self.update_search_history()
        
        self.running = True
        self.progress.pack(pady=10, fill='x')
        self.progress.start()
        
        self.search_btn.configure(
            text="🔄 Searching...",
            state="disabled"
        )
        
        self.status_label.configure(text=f"Searching for: {prompt}")
        
        threading.Thread(
            target=lambda: self.search_images(self.folder_path, prompt),
            daemon=True
        ).start()

    def update_search_history(self):
        self.history_listbox.configure(state="normal")
        self.history_listbox.delete("0.0", "end")
        
        for i, search in enumerate(reversed(self.search_history[-5:]), 1):
            self.history_listbox.insert("end", f"{i}. {search}\n")
        
        self.history_listbox.configure(state="disabled")

    def search_images(self, folder_path, prompt):
        try:
            start_time = time.time()
            
            # Update status
            self.after(0, lambda: self.status_label.configure(text="Loading images..."))
            images = load_images(folder_path)
            
            self.after(0, lambda: self.status_label.configure(text="Encoding search query..."))
            text_features = encode_text(prompt, self.model, self.device)
            
            self.after(0, lambda: self.status_label.configure(text="Analyzing images..."))
            results = []
            threshold = self.threshold_slider.get()
            
            for i, (img_path, img) in enumerate(images):
                img_features = encode_image(img, self.preprocess, self.model, self.device)
                similarity = (text_features @ img_features.T).item()
                
                if similarity >= threshold:
                    results.append((img_path, similarity))
                
                # Update progress
                progress_text = f"Processed {i+1}/{len(images)} images..."
                self.after(0, lambda t=progress_text: self.status_label.configure(text=t))
            
            results.sort(key=lambda x: x[1], reverse=True)
            final_results = results[:int(self.results_slider.get())]
            
            search_time = time.time() - start_time
            self.current_results = final_results
            
            self.after(0, lambda: self.show_search_results(final_results, search_time, prompt))
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            self.after(0, lambda: CTkMessagebox(
                title="Search Error",
                message=error_msg,
                icon="cancel"
            ))
            self.after(0, lambda: self.status_label.configure(text=error_msg))
        finally:
            self.after(0, self.reset_search_ui)

    def show_search_results(self, results, search_time, prompt):
        # Clear previous results
        for widget in self.results_scrollable.winfo_children():
            widget.destroy()
        
        if not results:
            self.show_no_results()
            return
        
        # Update results title
        self.results_title.configure(
            text=f"🖼️ Found {len(results)} matches for '{prompt}'"
        )
        
        # Display results based on view mode
        view_mode = self.view_mode.get().lower()
        self.display_results_in_mode(results, view_mode)
        
        # Add search statistics
        stats_frame = ctk.CTkFrame(self.results_scrollable, corner_radius=10)
        stats_frame.pack(pady=20, padx=10, fill="x")
        
        stats_text = f"⏱️ Search completed in {search_time:.2f}s • 🎯 Best match: {results[0][1]:.3f}"
        ctk.CTkLabel(
            stats_frame,
            text=stats_text,
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(pady=10)
        
        self.status_label.configure(text=f"Search completed: {len(results)} results in {search_time:.2f}s")

    def display_results_in_mode(self, results, mode):
        if mode == "grid":
            self.display_grid_results(results)
        elif mode == "list":
            self.display_list_results(results)
        else:  # detailed
            self.display_detailed_results(results)

    def display_grid_results(self, results):
        # Calculate columns based on window width
        window_width = self.winfo_width()
        cols = max(2, min(4, (window_width - 400) // 300))
        
        for i, (img_path, score) in enumerate(results):
            row = i // cols
            col = i % cols
            
            # Create image card
            card = self.create_image_card(img_path, score, size=(280, 280))
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Configure grid weights
            self.results_scrollable.grid_columnconfigure(col, weight=1)

    def display_list_results(self, results):
        for i, (img_path, score) in enumerate(results):
            card = self.create_image_card(img_path, score, size=(100, 100), horizontal=True)
            card.pack(pady=5, padx=10, fill="x")

    def display_detailed_results(self, results):
        for i, (img_path, score) in enumerate(results):
            card = self.create_detailed_card(img_path, score)
            card.pack(pady=10, padx=10, fill="x")

    def create_image_card(self, img_path, score, size=(280, 280), horizontal=False):
        card = ctk.CTkFrame(self.results_scrollable, corner_radius=15)
        
        try:
            # Load and process image
            img = Image.open(img_path)
            img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
            img_tk = ctk.CTkImage(img, size=size)
            
            if horizontal:
                # Horizontal layout for list view
                card.grid_columnconfigure(1, weight=1)
                
                img_label = ctk.CTkLabel(card, image=img_tk, text="")
                img_label.grid(row=0, column=0, padx=15, pady=15, sticky="w")
                
                info_frame = ctk.CTkFrame(card, fg_color="transparent")
                info_frame.grid(row=0, column=1, padx=15, pady=15, sticky="ew")
                
                self.add_image_info(info_frame, img_path, score)
                self.add_action_buttons(info_frame, img_path, horizontal=True)
            else:
                # Vertical layout for grid view
                img_label = ctk.CTkLabel(card, image=img_tk, text="")
                img_label.pack(padx=15, pady=(15, 5))
                
                info_frame = ctk.CTkFrame(card, fg_color="transparent")
                info_frame.pack(padx=15, pady=(0, 15), fill="x")
                
                self.add_image_info(info_frame, img_path, score)
                self.add_action_buttons(info_frame, img_path)
            
            # Hover effects
            self.add_hover_effects(card, img_label)
            
        except Exception as e:
            # Error card
            error_label = ctk.CTkLabel(
                card,
                text=f"❌ Error loading image\n{os.path.basename(img_path)}",
                font=ctk.CTkFont(size=12)
            )
            error_label.pack(pady=20)
        
        return card

    def create_detailed_card(self, img_path, score):
        card = ctk.CTkFrame(self.results_scrollable, corner_radius=15)
        
        try:
            # Get image info
            img = Image.open(img_path)
            width, height = img.size
            file_size = os.path.getsize(img_path)
            
            # Thumbnail
            thumb = ImageOps.fit(img, (150, 150), Image.Resampling.LANCZOS)
            img_tk = ctk.CTkImage(thumb, size=(150, 150))
            
            # Layout
            main_frame = ctk.CTkFrame(card, fg_color="transparent")
            main_frame.pack(padx=20, pady=20, fill="x")
            main_frame.grid_columnconfigure(1, weight=1)
            
            # Image thumbnail
            img_label = ctk.CTkLabel(main_frame, image=img_tk, text="")
            img_label.grid(row=0, column=0, padx=(0, 20), sticky="nw")
            
            # Detailed info
            info_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
            info_frame.grid(row=0, column=1, sticky="ew")
            
            # File name
            name_label = ctk.CTkLabel(
                info_frame,
                text=os.path.basename(img_path),
                font=ctk.CTkFont(size=16, weight="bold")
            )
            name_label.pack(anchor="w", pady=(0, 5))
            
            # Similarity score
            score_label = ctk.CTkLabel(
                info_frame,
                text=f"🎯 Similarity: {score:.3f}",
                font=ctk.CTkFont(size=14),
                text_color=self.get_score_color(score)
            )
            score_label.pack(anchor="w", pady=2)
            
            # Image dimensions
            dims_label = ctk.CTkLabel(
                info_frame,
                text=f"📐 Dimensions: {width}x{height}",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            )
            dims_label.pack(anchor="w", pady=2)
            
            # File size
            size_label = ctk.CTkLabel(
                info_frame,
                text=f"💾 Size: {self.format_file_size(file_size)}",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            )
            size_label.pack(anchor="w", pady=2)
            
            # Action buttons
            self.add_action_buttons(info_frame, img_path, detailed=True)
            
        except Exception as e:
            error_label = ctk.CTkLabel(
                card,
                text=f"❌ Error loading detailed view: {str(e)}",
                font=ctk.CTkFont(size=12)
            )
            error_label.pack(pady=20)
        
        return card

    def add_image_info(self, parent, img_path, score):
        # File name
        name_label = ctk.CTkLabel(
            parent,
            text=os.path.basename(img_path),
            font=ctk.CTkFont(size=12, weight="bold"),
            wraplength=200
        )
        name_label.pack(pady=(0, 5))
        
        # Score with color coding
        score_color = self.get_score_color(score)
        score_label = ctk.CTkLabel(
            parent,
            text=f"🎯 {score:.3f}",
            font=ctk.CTkFont(size=11),
            text_color=score_color
        )
        score_label.pack()

    def add_action_buttons(self, parent, img_path, horizontal=False, detailed=False):
        if detailed:
            button_frame = ctk.CTkFrame(parent, fg_color="transparent")
            button_frame.pack(pady=(10, 0), fill="x")
        else:
            button_frame = ctk.CTkFrame(parent, fg_color="transparent")
            button_frame.pack(pady=(5, 0))
        
        # Save button
        save_btn = ctk.CTkButton(
            button_frame,
            text="💾" if not detailed else "💾 Save",
            width=35 if not detailed else 80,
            height=25,
            command=lambda: self.save_image(img_path),
            font=ctk.CTkFont(size=10)
        )
        save_btn.pack(side="left", padx=(0, 5))
        
        # View button
        view_btn = ctk.CTkButton(
            button_frame,
            text="👁️" if not detailed else "👁️ View",
            width=35 if not detailed else 80,
            height=25,
            command=lambda: self.view_image(img_path),
            font=ctk.CTkFont(size=10)
        )
        view_btn.pack(side="left", padx=2)

    def add_hover_effects(self, card, img_label):
        def on_enter(e):
            card.configure(border_width=2, border_color=self.colors['primary'])
            
        def on_leave(e):
            card.configure(border_width=0)
            
        def on_click(e):
            # Get image path from card (you might need to store this)
            pass
            
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        img_label.bind("<Button-1>", on_click)

    def get_score_color(self, score):
        if score > 0.8:
            return self.colors['success']
        elif score > 0.6:
            return self.colors['accent']
        elif score > 0.4:
            return self.colors['warning']
        else:
            return self.colors['danger']

    def format_file_size(self, size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"

    def show_no_results(self):
        no_results_frame = ctk.CTkFrame(self.results_scrollable, corner_radius=15)
        no_results_frame.pack(pady=50, padx=50, fill="both", expand=True)
        
        # Icon
        icon_label = ctk.CTkLabel(
            no_results_frame,
            text="🔍",
            font=ctk.CTkFont(size=60)
        )
        icon_label.pack(pady=(30, 15))
        
        # Message
        title_label = ctk.CTkLabel(
            no_results_frame,
            text="No matching images found",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=5)
        
        subtitle_label = ctk.CTkLabel(
            no_results_frame,
            text="Try adjusting your search terms or similarity threshold",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        subtitle_label.pack(pady=(0, 30))

    def view_image(self, img_path):
        # Create a popup window to view the full image
        popup = ctk.CTkToplevel(self)
        popup.title(f"📷 {os.path.basename(img_path)}")
        popup.geometry("800x600")
        
        try:
            img = Image.open(img_path)
            # Resize to fit window while maintaining aspect ratio
            img.thumbnail((750, 550), Image.Resampling.LANCZOS)
            img_tk = ctk.CTkImage(img, size=img.size)
            
            img_label = ctk.CTkLabel(popup, image=img_tk, text="")
            img_label.pack(padx=20, pady=20, expand=True)
            
        except Exception as e:
            error_label = ctk.CTkLabel(
                popup,
                text=f"Error loading image: {str(e)}",
                font=ctk.CTkFont(size=14)
            )
            error_label.pack(pady=50)

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
                CTkMessagebox(
                    title="Success",
                    message="Image saved successfully!",
                    icon="check"
                )
            except Exception as e:
                CTkMessagebox(
                    title="Error",
                    message=f"Failed to save image: {str(e)}",
                    icon="cancel"
                )

    def export_results(self):
        if not self.current_results:
            CTkMessagebox(
                title="No Results",
                message="No search results to export!",
                icon="warning"
            )
            return
        
        # Export results to a text file
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("AI Image Search Results\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i, (img_path, score) in enumerate(self.current_results, 1):
                        f.write(f"{i}. {os.path.basename(img_path)}\n")
                        f.write(f"   Path: {img_path}\n")
                        f.write(f"   Similarity: {score:.6f}\n\n")
                
                CTkMessagebox(
                    title="Export Complete",
                    message="Results exported successfully!",
                    icon="check"
                )
            except Exception as e:
                CTkMessagebox(
                    title="Export Error",
                    message=f"Failed to export results: {str(e)}",
                    icon="cancel"
                )

    def clear_results(self):
        for widget in self.results_scrollable.winfo_children():
            widget.destroy()
        
        self.current_results = []
        self.results_title.configure(text="🖼️ Search Results")
        self.show_welcome_message()
        
        CTkMessagebox(
            title="Cleared",
            message="Results cleared successfully!",
            icon="info"
        )

    def reset_search_ui(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.running = False
        
        self.search_btn.configure(
            text="🚀 Search Images",
            state="normal"
        )


# Helper functions (keep these outside the class)
def load_images(folder_path):
    images = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')
    
    for file in os.listdir(folder_path):
        if file.lower().endswith(supported_formats):
            image_path = os.path.join(folder_path, file)
            try:
                img = Image.open(image_path).convert("RGB")
                images.append((image_path, img))
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