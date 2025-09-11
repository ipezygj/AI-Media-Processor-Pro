# ---------- BEGIN: app_ui.py (Fixed Crash on Start) ----------
"""
AI Media Processor Pro - Full UI (Revised and Patched)
Includes fixes for UI state management, settings persistence, mutually exclusive export
options, dependent UI controls, and added Whisper model selection.
** transcription progress updated to show an indeterminate bar. **
** Fixed a crash when starting processing due to incorrect state change on a tab frame. **
"""

import sys
print(f"--- SCRIPT IS RUNNING WITH: {sys.executable} ---")

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import threading
import os
from pathlib import Path
import json

# Import the backend logic
import processing_logic

CONFIG_FILE = "settings.json"
DEFAULT_SETTINGS = {
    "source_path": "",
    "output_path": str(Path.home() / "Downloads"),
    "stem_volumes": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0},
    "pitch_shift": 0,
    "speed_multiplier": 1.0,
    "normalize_volume": False,
    "generate_lyrics": True,
    "whisper_model": "large-v3",
    "karaoke_styles": {
        "font_name": "Arial",
        "font_size": 30,
        "upcoming_color": "#FFFFFF",
        "highlight_color": "#FF69B4",
        "outline_color": "#000000",
        "shadow_color": "#000000"
    },
    "export_mode": "Video", # Video, Audio Only, Stems Only
    "export_format": "mp3",
    "stems_to_export": {"vocals": True, "drums": False, "bass": False, "other": False},
    "appearance_mode": "Dark",
    "color_theme": "blue",
    "window_geometry": "900x950"
}

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.settings = self._load_settings()

        # --- Window Setup ---
        self.title("AI Media Processor Pro")
        self.geometry(self.settings.get("window_geometry", DEFAULT_SETTINGS["window_geometry"]))
        ctk.set_appearance_mode(self.settings.get("appearance_mode", DEFAULT_SETTINGS["appearance_mode"]))
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1) # Allow log to expand

        # --- State Variables ---
        self.cancel_flag = threading.Event()
        self.processing_thread = None
        self.interactive_widgets = []

        # --- Widgets ---
        self.create_widgets()
        self._create_context_menu()
        self.load_ui_from_settings()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _load_settings(self):
        """Loads settings from config file, filling gaps with defaults."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded_settings = json.load(f)
                # Merge loaded settings with defaults to ensure all keys exist
                settings = DEFAULT_SETTINGS.copy()
                settings.update(loaded_settings)
                return settings
            except (IOError, json.JSONDecodeError):
                return DEFAULT_SETTINGS.copy()
        return DEFAULT_SETTINGS.copy()

    # ----------------- CONTEXT MENU -----------------
    def _create_context_menu(self):
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Cut", command=lambda: self.focus_get().event_generate('<<Cut>>'))
        self.context_menu.add_command(label="Copy", command=lambda: self.focus_get().event_generate('<<Copy>>'))
        self.context_menu.add_command(label="Paste", command=lambda: self.focus_get().event_generate('<<Paste>>'))

    def _show_context_menu(self, event):
        event.widget.focus()
        self.context_menu.tk_popup(event.x_root, event.y_root)

    # ----------------- LABEL/UI UPDATES -----------------
    def _update_speed_label(self, value):
        self.speed_slider_label.configure(text=f"{float(value):.2f}x")
    
    def _update_mixer_label(self, stem, value):
        getattr(self, f"{stem}_mixer_label").configure(text=f"{int(float(value) * 100)}%")
        
    def _update_pitch_label(self, value):
        semitones = int(float(value))
        self.pitch_slider_label.configure(text=f"{'+' if semitones > 0 else ''}{semitones} st")
        
    def _update_font_size_label(self, value):
        self.font_size_label.configure(text=f"{int(float(value))}pt")
        
    def _pick_color(self, color_key, preview_widget):
        # Initial color should be the one currently stored
        initial_color = self.settings["karaoke_styles"][color_key]
        color_code = colorchooser.askcolor(initialcolor=initial_color, title=f"Choose {color_key.replace('_', ' ')} color")
        if color_code and color_code[1]:
            self.settings["karaoke_styles"][color_key] = color_code[1]
            preview_widget.configure(fg_color=color_code[1])

    def _on_export_mode_change(self, mode):
        """Enable/disable relevant export options based on the selected mode."""
        is_audio_mode = (mode == "Audio Only")
        is_stems_mode = (mode == "Stems Only")
        
        # Enable format menu for Audio or Stems export
        self.format_menu.configure(state=tk.NORMAL if (is_audio_mode or is_stems_mode) else tk.DISABLED)
        
        # Enable stem checkboxes only for Stems export
        if is_stems_mode and not any(self.stem_vars[s].get() for s in self.stem_vars):
             for var in self.stem_vars.values():
                var.set(True) # Pre-check all when switching to stem mode if none are checked

        for checkbox in self.stem_checkboxes.values():
            checkbox.configure(state=tk.NORMAL if is_stems_mode else tk.DISABLED)

    # ----------------- FILE PICKERS -----------------
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select media file",
            filetypes=[("Media files", "*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.flac"), ("All files", "*.*")]
        )
        if file_path:
            self.entry_source.delete(0, tk.END)
            self.entry_source.insert(0, file_path)

    def browse_output_folder(self):
        folder_path = filedialog.askdirectory(title="Select output folder", initialdir=self.entry_output_path.get())
        if folder_path:
            self.entry_output_path.delete(0, tk.END)
            self.entry_output_path.insert(0, folder_path)

    # ----------------- MAIN WIDGETS -----------------
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
        self.interactive_widgets.extend([self.entry_source, self.browse_button, self.entry_output_path, self.browse_output_button])

        # --- Options Tabs ---
        self.options_tab_view = ctk.CTkTabview(self, anchor="w")
        self.options_tab_view.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        self.options_tab_view.add("Main Mixer")
        self.options_tab_view.add("Audio Effects")
        self.options_tab_view.add("Karaoke")
        self.options_tab_view.add("Exports")
        self.interactive_widgets.append(self.options_tab_view)

        # ---- Main Mixer Tab ----
        mixer_tab = self.options_tab_view.tab("Main Mixer")
        stems = ["vocals", "drums", "bass", "other"]
        for i, stem in enumerate(stems):
            ctk.CTkLabel(mixer_tab, text=stem.capitalize()).grid(row=i, column=0, padx=10, pady=5, sticky="w")
            slider = ctk.CTkSlider(mixer_tab, from_=0, to=2, number_of_steps=200, command=lambda v, s=stem: self._update_mixer_label(s, v))
            slider.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
            setattr(self, f"{stem}_slider", slider)
            lbl = ctk.CTkLabel(mixer_tab, text="100%", width=40)
            lbl.grid(row=i, column=2, padx=10, pady=5)
            setattr(self, f"{stem}_mixer_label", lbl)
            self.interactive_widgets.append(slider)

        # ---- Audio Effects Tab ----
        effects_tab = self.options_tab_view.tab("Audio Effects")
        ctk.CTkLabel(effects_tab, text="Pitch Shift (semitones)").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.pitch_slider = ctk.CTkSlider(effects_tab, from_=-12, to=12, number_of_steps=24, command=self._update_pitch_label)
        self.pitch_slider.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.pitch_slider_label = ctk.CTkLabel(effects_tab, text="0 st", width=40)
        self.pitch_slider_label.grid(row=0, column=2, padx=10, pady=5)
        self.interactive_widgets.append(self.pitch_slider)

        ctk.CTkLabel(effects_tab, text="Speed Multiplier").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.speed_slider = ctk.CTkSlider(effects_tab, from_=0.5, to=2.0, number_of_steps=150, command=self._update_speed_label)
        self.speed_slider.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.speed_slider_label = ctk.CTkLabel(effects_tab, text="1.00x", width=40)
        self.speed_slider_label.grid(row=1, column=2, padx=10, pady=5)
        self.interactive_widgets.append(self.speed_slider)
        
        self.normalize_var = tk.BooleanVar()
        self.normalize_checkbox = ctk.CTkCheckBox(effects_tab, text="Normalize Volume (Loudness)", variable=self.normalize_var)
        self.normalize_checkbox.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="w")
        self.interactive_widgets.append(self.normalize_checkbox)

        # ---- Karaoke Tab ----
        karaoke_tab = self.options_tab_view.tab("Karaoke")
        self.lyrics_var = tk.BooleanVar()
        self.lyrics_checkbox = ctk.CTkCheckBox(karaoke_tab, text="Generate & Burn Karaoke Lyrics", variable=self.lyrics_var)
        self.lyrics_checkbox.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        self.interactive_widgets.append(self.lyrics_checkbox)
        
        ctk.CTkLabel(karaoke_tab, text="AI Model:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.whisper_model_var = tk.StringVar()
        self.whisper_model_menu = ctk.CTkOptionMenu(karaoke_tab, variable=self.whisper_model_var,
                                                    values=["tiny", "base", "small", "medium", "large-v3"])
        self.whisper_model_menu.grid(row=1, column=1, columnspan=2, padx=10, pady=5, sticky="ew")
        self.interactive_widgets.append(self.whisper_model_menu)

        ctk.CTkLabel(karaoke_tab, text="Font:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.font_entry = ctk.CTkEntry(karaoke_tab, placeholder_text="Arial")
        self.font_entry.grid(row=2, column=1, columnspan=2, padx=10, pady=5, sticky="ew")
        self.interactive_widgets.append(self.font_entry)

        ctk.CTkLabel(karaoke_tab, text="Font Size:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.font_size_slider = ctk.CTkSlider(karaoke_tab, from_=12, to=72, number_of_steps=60, command=self._update_font_size_label)
        self.font_size_slider.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        self.font_size_label = ctk.CTkLabel(karaoke_tab, text="30pt", width=40)
        self.font_size_label.grid(row=3, column=2, padx=10, pady=5)
        self.interactive_widgets.append(self.font_size_slider)

        karaoke_colors = {"upcoming_color": "Upcoming", "highlight_color": "Highlight", "outline_color": "Outline", "shadow_color": "Shadow"}
        for i, (key, text) in enumerate(karaoke_colors.items()):
            ctk.CTkLabel(karaoke_tab, text=f"{text} Color:").grid(row=4+i, column=0, padx=10, pady=5, sticky="w")
            preview = ctk.CTkLabel(karaoke_tab, text="", fg_color="black", width=80, height=20)
            preview.grid(row=4+i, column=1, padx=10, pady=5, sticky="w")
            button = ctk.CTkButton(karaoke_tab, text="Pick...", command=lambda k=key, p=preview: self._pick_color(k, p))
            button.grid(row=4+i, column=2, padx=10, pady=5)
            setattr(self, f"{key}_preview", preview)
            self.interactive_widgets.append(button)

        # ---- Exports Tab ----
        exports_tab = self.options_tab_view.tab("Exports")
        self.export_mode_var = tk.StringVar()
        self.export_mode_chooser = ctk.CTkSegmentedButton(exports_tab, variable=self.export_mode_var,
                                                         values=["Video", "Audio Only", "Stems Only"],
                                                         command=self._on_export_mode_change)
        self.export_mode_chooser.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        self.interactive_widgets.append(self.export_mode_chooser)

        ctk.CTkLabel(exports_tab, text="Audio Format:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.format_var = tk.StringVar()
        self.format_menu = ctk.CTkOptionMenu(exports_tab, variable=self.format_var, values=["mp3", "wav", "flac"])
        self.format_menu.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.interactive_widgets.append(self.format_menu)

        ctk.CTkLabel(exports_tab, text="Stems to Export:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.stem_vars = {stem: tk.BooleanVar() for stem in ["vocals", "drums", "bass", "other"]}
        self.stem_checkboxes = {}
        for i, stem in enumerate(self.stem_vars.keys()):
            cb = ctk.CTkCheckBox(exports_tab, text=stem.capitalize(), variable=self.stem_vars[stem])
            cb.grid(row=3+i, column=0, padx=20, pady=(0,5), sticky="w")
            self.stem_checkboxes[stem] = cb
            self.interactive_widgets.append(cb)

        # --- Progress Bar and Log ---
        self.progress_frame = ctk.CTkFrame(self)
        self.progress_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.progress_frame.grid_columnconfigure(0, weight=1)
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Ready to process.")
        self.progress_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew")

        self.log_textbox = ctk.CTkTextbox(self)
        self.log_textbox.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

        # --- Action Buttons ---
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.button_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.start_button = ctk.CTkButton(self.button_frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.cancel_button = ctk.CTkButton(self.button_frame, text="Cancel", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
    # ----------------- PROCESSING LOGIC -----------------
    def start_processing(self):
        source_path = self.entry_source.get()
        output_dir = self.entry_output_path.get()
        
        if not source_path or not output_dir:
            messagebox.showerror("Input Error", "Please provide a source URL/file and an output folder.")
            return
        if not os.path.isdir(output_dir):
            messagebox.showerror("Input Error", "The specified output folder does not exist.")
            return

        self.save_ui_to_settings() # Save settings before starting
        self.log_textbox.delete("1.0", tk.END)
        self.update_log("Starting processing...\n")
        self.cancel_flag.clear()
        self.toggle_ui_state(is_processing=True)
        
        self.processing_thread = threading.Thread(target=self._processing_thread, daemon=True)
        self.processing_thread.start()

    def cancel_processing(self):
        self.update_log("Cancellation signal sent. Please wait for the current step to finish...\n")
        self.cancel_flag.set()
        self.cancel_button.configure(state=tk.DISABLED, text="Cancelling...")

    def _processing_thread(self):
        try:
            # All settings are now read from the self.settings dict
            s = self.settings
            processing_logic.process_media(
                source_path=s["source_path"],
                output_dir_base=s["output_path"],
                stem_volumes=s["stem_volumes"],
                pitch_shift=s["pitch_shift"],
                normalize_volume=s["normalize_volume"],
                speed_multiplier=s["speed_multiplier"],
                generate_lyrics=s["generate_lyrics"],
                whisper_model=s["whisper_model"],
                karaoke_styles=s["karaoke_styles"],
                cancel_flag=self.cancel_flag,
                progress_callback=self.update_progress,
                export_mode=s["export_mode"],
                export_format=s["export_format"],
                stems_to_export=[stem for stem, selected in s["stems_to_export"].items() if selected]
            )
        except processing_logic.CancelledError:
            self.update_log("Processing was successfully cancelled by the user.\n")
        except processing_logic.ProcessingError as e:
            self.update_log(f"\n❌ PROCESSING ERROR:\n{e}\n\nDetails:\n{e.details}\n")
        except Exception as e:
            self.update_log(f"\n❌ An unexpected error occurred: {type(e).__name__}: {e}\n")
        finally:
            self.toggle_ui_state(is_processing=False)
            self.update_progress("Ready to process.", 0)

    def update_progress(self, message, percentage):
        """
        Updates the progress bar and label.
        If percentage is negative, enters indeterminate mode for long-running tasks.
        """
        # This function is called from a different thread, so we use after()
        # to ensure UI updates happen on the main thread.
        self.after(0, self._update_progress_ui, message, percentage)

    def _update_progress_ui(self, message, percentage):
        self.update_log(f"[{'BUSY' if percentage < 0 else str(int(percentage))+'%'}] {message}\n")
        self.progress_label.configure(text=message)

        if percentage < 0:
            # Enter indeterminate mode
            if self.progress_bar.cget("mode") == "determinate":
                self.progress_bar.configure(mode='indeterminate')
                self.progress_bar.start()
        else:
            # Enter determinate mode
            if self.progress_bar.cget("mode") == "indeterminate":
                self.progress_bar.stop()
                self.progress_bar.configure(mode='determinate')
            self.progress_bar.set(percentage / 100.0)

    def update_log(self, message):
        self.log_textbox.insert(tk.END, message)
        self.log_textbox.see(tk.END)

    def toggle_ui_state(self, is_processing):
        state = tk.DISABLED if is_processing else tk.NORMAL
        for widget in self.interactive_widgets:
             widget.configure(state=state)

        # THE BUGGY LOOP WAS HERE AND HAS BEEN REMOVED.
        # The widgets inside the tabs are already in self.interactive_widgets,
        # so they are disabled correctly by the loop above.

        self.start_button.configure(state=state)
        self.cancel_button.configure(state=tk.NORMAL if is_processing else tk.DISABLED, text="Cancel")

    # ----------------- SETTINGS HANDLING -----------------
    def save_ui_to_settings(self):
        s = self.settings
        s["source_path"] = self.entry_source.get()
        s["output_path"] = self.entry_output_path.get()
        s["stem_volumes"] = {stem: getattr(self, f"{stem}_slider").get() for stem in ["vocals", "drums", "bass", "other"]}
        s["pitch_shift"] = self.pitch_slider.get()
        s["speed_multiplier"] = self.speed_slider.get()
        s["normalize_volume"] = self.normalize_var.get()
        s["generate_lyrics"] = self.lyrics_var.get()
        s["whisper_model"] = self.whisper_model_var.get()
        s["karaoke_styles"]["font_name"] = self.font_entry.get()
        s["karaoke_styles"]["font_size"] = int(self.font_size_slider.get())
        s["export_mode"] = self.export_mode_var.get()
        s["export_format"] = self.format_var.get()
        s["stems_to_export"] = {stem: var.get() for stem, var in self.stem_vars.items()}
        s["appearance_mode"] = ctk.get_appearance_mode()
        s["window_geometry"] = self.geometry()

    def load_ui_from_settings(self):
        s = self.settings
        self.entry_source.insert(0, s["source_path"])
        self.entry_output_path.insert(0, s["output_path"])
        for stem, vol in s["stem_volumes"].items():
            getattr(self, f"{stem}_slider").set(vol)
            self._update_mixer_label(stem, vol)
        self.pitch_slider.set(s["pitch_shift"])
        self._update_pitch_label(s["pitch_shift"])
        self.speed_slider.set(s["speed_multiplier"])
        self._update_speed_label(s["speed_multiplier"])
        self.normalize_var.set(s["normalize_volume"])
        self.lyrics_var.set(s["generate_lyrics"])
        self.whisper_model_var.set(s.get("whisper_model", "large-v3"))
        
        ks = s["karaoke_styles"]
        self.font_entry.insert(0, ks["font_name"])
        self.font_size_slider.set(ks["font_size"])
        self._update_font_size_label(ks["font_size"])
        
        self.upcoming_color_preview.configure(fg_color=ks["upcoming_color"])
        self.highlight_color_preview.configure(fg_color=ks["highlight_color"])
        self.outline_color_preview.configure(fg_color=ks.get("outline_color", "#000000"))
        self.shadow_color_preview.configure(fg_color=ks.get("shadow_color", "#000000"))

        self.export_mode_var.set(s["export_mode"])
        self.format_var.set(s["export_format"])
        for stem, selected in s["stems_to_export"].items():
            self.stem_vars[stem].set(selected)
        
        self._on_export_mode_change(s["export_mode"])

    def on_closing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            if messagebox.askyesno("Confirm Exit", "Processing is still in progress. Are you sure you want to exit?"):
                self.cancel_flag.set()
                self.destroy()
        else:
            self.save_ui_to_settings()
            try:
                with open(CONFIG_FILE, "w") as f:
                    json.dump(self.settings, f, indent=4)
            except IOError as e:
                print(f"Failed to save settings: {e}")
            self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
# ---------- END: app_ui.py (Fixed Crash on Start) ----------
