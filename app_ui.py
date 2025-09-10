# File: app_ui.py

import sys
print(f"--- SCRIPT IS RUNNING WITH: {sys.executable} ---")

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import threading
import os
import subprocess
from pathlib import Path
import json
import re
from functools import partial

# Import the backend logic
import processing_logic

CONFIG_FILE = "settings.json"

# --- Helper function to load settings before app creation ---
def _load_initial_settings():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError):
        pass
    return {}

class App(ctk.CTk):
    def __init__(self, initial_settings):
        super().__init__()

        # --- Window Setup ---
        self.title("AI Media Processor Pro")
        self.geometry("800x850")
        
        ctk.set_default_color_theme(initial_settings.get("color_theme", "sweetkind"))
        ctk.set_appearance_mode(initial_settings.get("appearance_mode", "Dark"))

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # --- State Variables ---
        self.queue = []
        self.current_job_id = None
        self.is_processing = False
        self.cancel_flag = threading.Event()
        self.last_successful_output_path = None
        self.karaoke_upcoming_color = "#FFFFFF"
        self.karaoke_highlight_color = "#FF69B4"

        # --- Widgets & Initial Setup ---
        self.create_widgets()
        self._create_context_menu()
        self.load_ui_settings(initial_settings)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_context_menu(self):
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Cut", command=lambda: self.focus_get().event_generate('<<Cut>>'))
        self.context_menu.add_command(label="Copy", command=lambda: self.focus_get().event_generate('<<Copy>>'))
        self.context_menu.add_command(label="Paste", command=lambda: self.focus_get().event_generate('<<Paste>>'))
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Select All", command=lambda: self.focus_get().event_generate('<<SelectAll>>'))

    def _show_context_menu(self, event):
        event.widget.focus()
        self.context_menu.tk_popup(event.x_root, event.y_root)

    def _update_speed_label(self, value):
        self.speed_slider_label.configure(text=f"{value:.2f}x")
    
    def _update_mixer_label(self, stem, value):
        getattr(self, f"{stem}_mixer_label").configure(text=f"{int(value * 100)}%")
        
    def _update_pitch_label(self, value):
        semitones = int(value)
        self.pitch_slider_label.configure(text=f"{'+' if semitones > 0 else ''}{semitones} st")
        
    def _update_font_size_label(self, value):
        self.font_size_label.configure(text=f"{int(value)}pt")
        
    def _pick_color(self, color_type):
        color_code = colorchooser.askcolor(title=f"Choose {color_type} color")
        if color_code and color_code[1]:
            if color_type == "upcoming":
                self.karaoke_upcoming_color = color_code[1]
                self.upcoming_color_preview.configure(fg_color=self.karaoke_upcoming_color)
            elif color_type == "highlight":
                self.karaoke_highlight_color = color_code[1]
                self.highlight_color_preview.configure(fg_color=self.karaoke_highlight_color)
                
    def _change_color_theme(self, new_theme: str):
        self.save_settings()
        messagebox.showinfo("Restart Required", f"Theme changed to '{new_theme}'.\nPlease restart the application to see the changes.")

    def create_widgets(self):
        # --- Top Frame (Input/Output) ---
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        self.top_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.top_frame, text="Source (URL or Local File):").grid(row=0, column=0, columnspan=3, padx=10, pady=(10,0), sticky="w")
        self.entry_source = ctk.CTkEntry(self.top_frame, placeholder_text="https://www.youtube.com/watch?v=...")
        self.entry_source.grid(row=1, column=0, columnspan=2, padx=(10, 5), pady=5, sticky="ew")
        self.entry_source.bind("<Button-3>", self._show_context_menu)
        self.browse_button = ctk.CTkButton(self.top_frame, text="Browse...", command=self.browse_file)
        self.browse_button.grid(row=1, column=2, padx=(5, 10), pady=5)
        
        ctk.CTkLabel(self.top_frame, text="Output Folder:").grid(row=2, column=0, padx=10, pady=(10, 0), sticky="w")
        self.entry_output_path = ctk.CTkEntry(self.top_frame)
        self.entry_output_path.grid(row=3, column=0, columnspan=2, padx=(10, 5), pady=5, sticky="ew")
        self.entry_output_path.bind("<Button-3>", self._show_context_menu)
        self.browse_output_button = ctk.CTkButton(self.top_frame, text="Browse...", command=self.browse_output_folder)
        self.browse_output_button.grid(row=3, column=2, padx=(5, 10), pady=10)

        # --- Options Tabs ---
        self.options_tab_view = ctk.CTkTabview(self, anchor="w")
        self.options_tab_view.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        self.options_tab_view.add("Main Mixer")
        self.options_tab_view.add("Audio Effects")
        self.options_tab_view.add("Karaoke")
        
        # Tab 1: Main Mixer
        self.mixer_tab = self.options_tab_view.tab("Main Mixer")
        self.mixer_tab.grid_columnconfigure(1, weight=1)
        self.stems = ["vocals", "drums", "bass", "other"]
        for i, stem in enumerate(self.stems):
            ctk.CTkLabel(self.mixer_tab, text=f"{stem.capitalize()}:").grid(row=i, column=0, padx=10, pady=10, sticky="w")
            slider = ctk.CTkSlider(self.mixer_tab, from_=0, to=1.5, number_of_steps=30, command=partial(self._update_mixer_label, stem))
            slider.set(1.0 if stem != "vocals" else 0.0)
            slider.grid(row=i, column=1, padx=10, pady=10, sticky="ew")
            setattr(self, f"{stem}_mixer_slider", slider)
            value_label = ctk.CTkLabel(self.mixer_tab, text=f"{int(slider.get() * 100)}%", width=40)
            value_label.grid(row=i, column=2, padx=10, pady=10, sticky="w")
            setattr(self, f"{stem}_mixer_label", value_label)

        # Tab 2: Audio Effects
        self.effects_tab = self.options_tab_view.tab("Audio Effects")
        self.effects_tab.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.effects_tab, text="Pitch Shift (Key):").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.pitch_slider = ctk.CTkSlider(self.effects_tab, from_=-12, to=12, number_of_steps=24, command=self._update_pitch_label)
        self.pitch_slider.set(0)
        self.pitch_slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.pitch_slider_label = ctk.CTkLabel(self.effects_tab, text="0 st", width=50)
        self.pitch_slider_label.grid(row=0, column=2, padx=10, pady=10, sticky="w")
        self.normalize_checkbox = ctk.CTkCheckBox(self.effects_tab, text="Normalize Volume (for consistent loudness)")
        self.normalize_checkbox.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        ctk.CTkLabel(self.effects_tab, text="Playback Speed:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.speed_slider = ctk.CTkSlider(self.effects_tab, from_=0.5, to=2.0, number_of_steps=15, command=self._update_speed_label)
        self.speed_slider.set(1.0)
        self.speed_slider.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        self.speed_slider_label = ctk.CTkLabel(self.effects_tab, text="1.00x", width=40)
        self.speed_slider_label.grid(row=2, column=2, padx=10, pady=10, sticky="w")

        # Tab 3: Karaoke
        self.karaoke_tab = self.options_tab_view.tab("Karaoke")
        self.karaoke_tab.grid_columnconfigure(1, weight=1)
        self.lyrics_checkbox = ctk.CTkCheckBox(self.karaoke_tab, text="🎤 Generate Karaoke Lyrics")
        self.lyrics_checkbox.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 0), sticky="w")
        # CHANGED: Updated warning label
        ctk.CTkLabel(self.karaoke_tab, text="(Slower, but more accurate default. Requires one-time download per model.)", text_color="gray", font=ctk.CTkFont(size=11)).grid(row=1, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="w")
        ctk.CTkLabel(self.karaoke_tab, text="AI Model:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.whisper_model_menu = ctk.CTkOptionMenu(self.karaoke_tab, values=["tiny", "base", "small", "medium"])
        self.whisper_model_menu.set("small") # CHANGED: Default to 'small' for better quality
        self.whisper_model_menu.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        self.style_frame = ctk.CTkFrame(self.karaoke_tab)
        self.style_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        self.style_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.style_frame, text="Font:").grid(row=0, column=0, padx=(10,5), pady=5, sticky="w")
        self.font_menu = ctk.CTkOptionMenu(self.style_frame, values=["Arial", "Comic Sans MS", "Impact", "Georgia"])
        self.font_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkLabel(self.style_frame, text="Font Size:").grid(row=1, column=0, padx=(10,5), pady=5, sticky="w")
        self.font_size_slider = ctk.CTkSlider(self.style_frame, from_=24, to=96, number_of_steps=72, command=self._update_font_size_label)
        self.font_size_slider.set(60)
        self.font_size_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.font_size_label = ctk.CTkLabel(self.style_frame, text="60pt", width=50)
        self.font_size_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        ctk.CTkButton(self.style_frame, text="Upcoming Text Color", command=lambda: self._pick_color("upcoming")).grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.upcoming_color_preview = ctk.CTkFrame(self.style_frame, fg_color=self.karaoke_upcoming_color, width=80, height=20, corner_radius=5, border_width=1, border_color="gray50")
        self.upcoming_color_preview.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ctk.CTkButton(self.style_frame, text="Highlight Text Color", command=lambda: self._pick_color("highlight")).grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.highlight_color_preview = ctk.CTkFrame(self.style_frame, fg_color=self.karaoke_highlight_color, width=80, height=20, corner_radius=5, border_width=1, border_color="gray50")
        self.highlight_color_preview.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # --- Queue Actions Frame ---
        self.queue_actions_frame = ctk.CTkFrame(self)
        self.queue_actions_frame.grid(row=2, column=0, padx=20, pady=0, sticky="ew")
        self.queue_actions_frame.grid_columnconfigure(0, weight=1)
        self.add_to_queue_button = ctk.CTkButton(self.queue_actions_frame, text="➕ Add Job to Queue", command=self.add_job_to_queue)
        self.add_to_queue_button.grid(row=0, column=0, padx=(5,5), pady=10, sticky="ew")
        self.clear_queue_button = ctk.CTkButton(self.queue_actions_frame, text="Clear All", width=100, command=self.clear_queue)
        self.clear_queue_button.grid(row=0, column=1, padx=(5,5), pady=10, sticky="e")

        # --- Main Content Tabs (Queue & Logs) ---
        self.main_content_tabs = ctk.CTkTabview(self, anchor="w")
        self.main_content_tabs.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")
        self.main_content_tabs.add("Processing Queue")
        self.main_content_tabs.add("Logs")
        
        self.queue_tab = self.main_content_tabs.tab("Processing Queue")
        self.queue_tab.grid_columnconfigure(0, weight=1); self.queue_tab.grid_rowconfigure(0, weight=1)
        self.queue_scroll_frame = ctk.CTkScrollableFrame(self.queue_tab, label_text="")
        self.queue_scroll_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.queue_scroll_frame.grid_columnconfigure(0, weight=1)
        
        self.log_tab = self.main_content_tabs.tab("Logs")
        self.log_tab.grid_columnconfigure(0, weight=1); self.log_tab.grid_rowconfigure(0, weight=1)
        self.log_textbox = ctk.CTkTextbox(self.log_tab, state="disabled", activate_scrollbars=True)
        self.log_textbox.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # --- Bottom Controls / Status Bar ---
        self.bottom_control_frame = ctk.CTkFrame(self)
        self.bottom_control_frame.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.bottom_control_frame.grid_columnconfigure(0, weight=1)

        # CHANGED: New dynamic status label
        self.status_label = ctk.CTkLabel(self.bottom_control_frame, text="Idle.", anchor="w")
        self.status_label.grid(row=0, column=0, columnspan=5, padx=10, pady=(5,0), sticky="w")

        # CHANGED: Progress bars are now linked to the new status label
        self.task_progress_bar = ctk.CTkProgressBar(self.bottom_control_frame)
        self.task_progress_bar.set(0)
        self.task_progress_bar.grid(row=1, column=0, columnspan=5, padx=10, pady=(0,5), sticky="ew")
        self.queue_progress_bar = ctk.CTkProgressBar(self.bottom_control_frame)
        self.queue_progress_bar.set(0)
        self.queue_progress_bar.grid(row=2, column=0, columnspan=5, padx=10, pady=(0,10), sticky="ew")

        # Action Buttons
        self.start_queue_button = ctk.CTkButton(self.bottom_control_frame, text="▶️ Start Queue", height=40, command=self.start_queue_thread)
        self.start_queue_button.grid(row=3, column=0, padx=(10,5), pady=5, sticky="ew")
        self.cancel_button = ctk.CTkButton(self.bottom_control_frame, text="🛑 Stop Processing", height=40, command=self.cancel_processing, fg_color="#D32F2F", hover_color="#B71C1C")
        self.open_folder_button = ctk.CTkButton(self.bottom_control_frame, text="📂 Open Output", command=self.open_output_folder, state="disabled")
        self.open_folder_button.grid(row=3, column=1, padx=(5,10), pady=5, sticky="e")
        ctk.CTkLabel(self.bottom_control_frame, text="Appearance:").grid(row=3, column=2, padx=(20,5), pady=5, sticky="w")
        self.appearance_menu = ctk.CTkOptionMenu(self.bottom_control_frame, values=["Dark", "Light", "System"], command=self.save_settings)
        self.appearance_menu.grid(row=3, column=3, padx=(0,10), pady=5, sticky="w")
        ctk.CTkLabel(self.bottom_control_frame, text="Color:").grid(row=3, column=4, padx=(10,5), pady=5, sticky="w")
        self.color_theme_menu = ctk.CTkOptionMenu(self.bottom_control_frame, values=["sweetkind", "blue", "dark-blue", "green"], command=self._change_color_theme)
        self.color_theme_menu.grid(row=3, column=5, padx=(0,10), pady=5, sticky="w")

    def log_message(self, message):
        # ... (unchanged)
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert(tk.END, message + "\n")
        self.log_textbox.see(tk.END)
        self.log_textbox.configure(state="disabled")
        
    def add_job_to_queue(self):
        # ... (unchanged)
        source = self.entry_source.get()
        if not self.validate_input(source): return
        if not self.entry_output_path.get():
            messagebox.showerror("Missing Info", "Please provide an output folder.")
            return
        karaoke_styles = {"font_name": self.font_menu.get(), "font_size": int(self.font_size_slider.get()), "upcoming_color": self.karaoke_upcoming_color, "highlight_color": self.karaoke_highlight_color}
        job = {"id": os.urandom(8).hex(), "source": source, "output_dir": self.entry_output_path.get(), "stem_volumes": {stem: getattr(self, f"{stem}_mixer_slider").get() for stem in self.stems}, "pitch_shift": int(self.pitch_slider.get()), "normalize_volume": self.normalize_checkbox.get(), "speed_multiplier": self.speed_slider.get(), "generate_lyrics": self.lyrics_checkbox.get(), "whisper_model": self.whisper_model_menu.get(), "karaoke_styles": karaoke_styles}
        self.queue.append(job)
        self._update_queue_display()
        self.entry_source.delete(0, tk.END)

    def start_queue_thread(self):
        # ... (unchanged)
        if self.is_processing:
            self.log_message("⚠️ Already processing. Please wait.")
            return
        if not self.queue:
            messagebox.showinfo("Queue Empty", "Please add one or more jobs to the queue first.")
            return
        self.is_processing = True
        self.cancel_flag.clear()
        self.log_textbox.configure(state="normal"); self.log_textbox.delete("1.0", tk.END); self.log_textbox.configure(state="disabled")
        self.set_ui_state(True)
        thread = threading.Thread(target=self._process_queue)
        thread.daemon = True
        thread.start()

    def _process_queue(self):
        # CHANGED: Logic for dual progress bars
        total_jobs = len(self.queue)
        jobs_completed = 0
        self.after(0, self.log_message, f"🚀 Starting queue with {total_jobs} job(s)...")

        try:
            while self.queue and not self.cancel_flag.is_set():
                job = self.queue[0]
                self.current_job_id = job["id"]
                
                self.after(0, self._update_queue_display)
                self.after(0, self.task_progress_bar.set, 0)
                self.after(0, self.queue_progress_bar.set, jobs_completed / total_jobs)
                self.after(0, self.log_message, f"\n--- Starting Job {jobs_completed + 1}/{total_jobs}: {Path(job['source']).name} ---")
                
                job_successful = False
                try:
                    processing_logic.process_media(
                        job["source"], job["output_dir"], job["stem_volumes"],
                        job["pitch_shift"], job["normalize_volume"],
                        job["speed_multiplier"], job["generate_lyrics"], job["whisper_model"],
                        job["karaoke_styles"], self.cancel_flag, self.update_progress)
                    job_successful = True
                except processing_logic.CancelledError:
                    self.after(0, self.log_message, "--- 🛑 Job cancelled by user. ---"); break
                except Exception as e:
                    self.after(0, self.log_message, f"--- ❌ Job '{Path(job['source']).name}' failed. Halting queue. ---")
                    import traceback
                    self.after(0, self.log_message, f"Error: {e}\n{traceback.format_exc()}"); break
                
                if job_successful:
                    self.queue.pop(0)
                    jobs_completed += 1
                    self.after(0, self._update_queue_display)
        finally:
            self.current_job_id = None
            self.is_processing = False
            self.after(0, self._update_queue_display)
            self.after(0, self.set_ui_state, False)
            if self.cancel_flag.is_set():
                self.after(0, self.status_label.configure, text="Queue stopped by user.")
                self.after(0, self.log_message, "\n--- Queue processing stopped by user. ---")
                self.queue.clear()
                self.after(0, self._update_queue_display)
            elif not self.queue:
                self.after(0, self.status_label.configure, text="Queue Complete!")
                self.after(0, self.log_message, "\n--- ✅ Queue processing complete! ---")
            else:
                self.after(0, self.status_label.configure, text=f"Queue stopped due to an error. {len(self.queue)} job(s) remaining.")
                self.after(0, self.log_message, f"\n--- ⚠️ Queue processing stopped due to an error. {len(self.queue)} job(s) remaining. ---")

    def set_ui_state(self, is_processing):
        # ... (unchanged)
        state = "disabled" if is_processing else "normal"
        widgets_to_change = [self.entry_source, self.browse_button, self.entry_output_path, self.browse_output_button, self.add_to_queue_button, self.clear_queue_button, self.pitch_slider, self.normalize_checkbox, self.speed_slider, self.lyrics_checkbox, self.whisper_model_menu, self.font_menu, self.font_size_slider]
        for stem in self.stems: widgets_to_change.append(getattr(self, f"{stem}_mixer_slider"))
        for child in self.style_frame.winfo_children():
            if isinstance(child, ctk.CTkButton): widgets_to_change.append(child)
        for widget in widgets_to_change:
            if hasattr(widget, 'configure'): widget.configure(state=state)
        if is_processing:
            self.start_queue_button.grid_remove()
            self.cancel_button.grid(row=3, column=0, padx=(10,5), pady=5, sticky="ew")
            self.cancel_button.configure(state="normal", text="🛑 Stop Processing")
            self.open_folder_button.configure(state="disabled")
        else:
            self.cancel_button.grid_remove()
            self.start_queue_button.grid()
            self.task_progress_bar.set(0)
            self.queue_progress_bar.set(0)
            self.status_label.configure(text="Idle.")
            if self.last_successful_output_path: self.open_folder_button.configure(state="normal")
    
    def _remove_job_from_queue(self, job_id):
        # ... (unchanged)
        if self.is_processing: return
        self.queue = [job for job in self.queue if job["id"] != job_id]
        self._update_queue_display()
    
    def clear_queue(self):
        # ... (unchanged)
        if self.is_processing: return
        if self.queue and messagebox.askyesno("Confirm", "Are you sure you want to clear the entire queue?"):
            self.queue.clear()
            self._update_queue_display()

    def _update_queue_display(self):
        # ... (unchanged)
        for widget in self.queue_scroll_frame.winfo_children(): widget.destroy()
        for i, job in enumerate(self.queue):
            fg_color = ("#C9E4E8", "#2E3B42") if job["id"] == self.current_job_id else ("gray85", "gray20")
            job_frame = ctk.CTkFrame(self.queue_scroll_frame, fg_color=fg_color)
            job_frame.grid(row=i, column=0, padx=5, pady=5, sticky="ew")
            job_frame.grid_columnconfigure(0, weight=1)
            source_text = Path(job["source"]).name if os.path.exists(job["source"]) else job["source"]
            label_text = f"▶ PROCESSING: {source_text}" if job["id"] == self.current_job_id else f"{i+1}. {source_text}"
            label = ctk.CTkLabel(job_frame, text=label_text, wraplength=500, justify="left")
            label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
            remove_button = ctk.CTkButton(job_frame, text="Remove", width=70, command=partial(self._remove_job_from_queue, job["id"]))
            remove_button.grid(row=0, column=1, padx=10, pady=5, sticky="e")
            if self.is_processing: remove_button.configure(state="disabled")

    def cancel_processing(self):
        # ... (unchanged)
        if self.is_processing and messagebox.askyesno("Confirm Stop", "Stop the current job and clear the queue?"):
            self.log_message("🛑 Sending stop signal... Finishing current step...")
            self.cancel_flag.set()
            self.cancel_button.configure(state="disabled", text="Stopping...")
            
    def update_progress(self, message, percentage): self.after(0, self._update_gui, message, percentage)
    
    def _update_gui(self, message, percentage):
        # CHANGED: Update the new status label
        self.status_label.configure(text=f"Current Task: {message}")
        self.task_progress_bar.set(percentage / 100)
        if "✅ Success!" in message:
            path_str = message.split("saved to ")[-1]
            self.last_successful_output_path = os.path.dirname(path_str)
            self.open_folder_button.configure(state="normal")
            
    def validate_input(self, source):
        # ... (unchanged)
        if re.match(r'^(https|http)?:\/\/(www\.)?(youtube\.com|youtu\.be)\/.+$', source) or os.path.exists(source): return True
        self.log_message("❌ Error: Input is not a valid YouTube URL or an existing local file.")
        messagebox.showerror("Invalid Input", "The input must be a valid YouTube URL or a local file path that exists.")
        return False
        
    def browse_file(self):
        # ... (unchanged)
        if filepath := filedialog.askopenfilename(filetypes=(("Media Files", "*.mp4 *.mkv *.mov *.avi *.mp3 *.wav *.flac"), ("All files", "*.*"))): 
            self.entry_source.delete(0, tk.END); self.entry_source.insert(0, filepath)
            
    def browse_output_folder(self):
        # ... (unchanged)
        if folder_path := filedialog.askdirectory(): 
            self.entry_output_path.delete(0, tk.END); self.entry_output_path.insert(0, folder_path)
            
    def open_output_folder(self):
        # ... (unchanged)
        if self.last_successful_output_path and os.path.exists(self.last_successful_output_path):
            try:
                if sys.platform == "win32": os.startfile(self.last_successful_output_path)
                elif sys.platform == "darwin": subprocess.run(["open", self.last_successful_output_path], check=True)
                else: subprocess.run(["xdg-open", self.last_successful_output_path], check=True)
            except Exception as e: self.log_message(f"⚠️ Could not open folder: {e}")
        else: self.log_message("⚠️ Output folder not found.")
            
    def load_ui_settings(self, settings):
        # ... (unchanged)
        self.entry_output_path.insert(0, settings.get("output_path", os.path.join(Path.home(), 'AIVideoProcessor')))
        self.appearance_menu.set(settings.get("appearance_mode", "Dark"))
        self.color_theme_menu.set(settings.get("color_theme", "sweetkind"))

    def save_settings(self, new_appearance_mode=None):
        # ... (unchanged)
        settings = {"output_path": self.entry_output_path.get(), "color_theme": self.color_theme_menu.get(), "appearance_mode": self.appearance_menu.get() if new_appearance_mode is None else new_appearance_mode}
        if new_appearance_mode:
             ctk.set_appearance_mode(new_appearance_mode)
        try:
            with open(CONFIG_FILE, 'w') as f: json.dump(settings, f, indent=4)
        except IOError as e: print(f"Could not save settings: {e}")

    def on_closing(self):
        # ... (unchanged)
        self.save_settings()
        if self.is_processing and messagebox.askyesno("Confirm Exit", "A process is still running. Stop the queue and exit?"):
            self.cancel_flag.set()
            self.destroy()
        elif not self.is_processing:
            self.destroy()

if __name__ == "__main__":
    settings = _load_initial_settings()
    app = App(initial_settings=settings)
    app.mainloop()