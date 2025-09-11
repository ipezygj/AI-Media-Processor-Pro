# ---------- BEGIN: processing_logic.py (Fixed) ----------
import os
import sys
import subprocess
import shutil
import math
import re
import yt_dlp
import ffmpeg
import torch
from pathlib import Path
import stable_whisper

# --- Configuration ---
CHUNK_DURATION_SECONDS = 300
LYRIC_PRE_DISPLAY_OFFSET_SECONDS = 0.5

class CancelledError(Exception):
    pass

class ProcessingError(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or "No additional details provided."

def sanitize_filename(name):
    """Removes characters that are invalid for file names."""
    return re.sub(r'[\\/*?:"<>|]', "", name)

def check_cancel(cancel_flag):
    """Checks if the user has requested to cancel processing."""
    if cancel_flag.is_set():
        raise CancelledError("Processing was cancelled by the user.")

def hex_to_ass_color(hex_color):
    """Converts a web hex color (#RRGGBB) to an ASS format color (&HBBGGRR&)."""
    if not isinstance(hex_color, str) or not re.match(r'^#[0-9a-fA-F]{6}$', hex_color):
        return "&H00FFFFFF&"  # Default to white on error
    return f"&H{hex_color[5:7]}{hex_color[3:5]}{hex_color[1:3]}&".upper()

def _generate_karaoke_subtitles(transcription_result, output_path, styles):
    """
    (REWRITTEN) Generates a correctly formatted Advanced SubStation Alpha (.ass)
    subtitle file for a word-by-word karaoke highlighting effect.
    """
    highlight_color = hex_to_ass_color(styles.get("highlight_color", "#FFFF00"))
    upcoming_color = hex_to_ass_color(styles.get("upcoming_color", "#FFFFFF"))
    outline_color = hex_to_ass_color(styles.get("outline_color", "#000000"))
    shadow_color = hex_to_ass_color(styles.get("shadow_color", "#000000"))
    font_name = styles.get("font_name", "Arial")
    font_size = styles.get("font_size", 30)

    with open(output_path, "w", encoding="utf-8-sig") as f:
        # --- ASS Header ---
        f.write("[Script Info]\nTitle: Karaoke Lyrics\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
                "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
                "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        # --- The single, correctly configured style for karaoke ---
        # PrimaryColour is the color AFTER highlighting (the "fill" color).
        # SecondaryColour is the color BEFORE highlighting (the "upcoming" color).
        # The \k tag transitions from Secondary to Primary color over its duration.
        f.write(f"Style: Karaoke,{font_name},{font_size},{highlight_color},{upcoming_color},{outline_color},{shadow_color},"
                "-1,0,0,0,100,100,0,0,1,2,2,2,10,10,20,1\n\n")
        
        # --- Events (The actual lyrics) ---
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for segment in transcription_result.segments:
            # Line-level timing
            start_s = max(0, segment.start - LYRIC_PRE_DISPLAY_OFFSET_SECONDS)
            start_time = f"{int(start_s//3600)}:{int(start_s%3600//60):02}:{int(start_s%60):02}.{int(start_s%1*100):02}"
            end_time = f"{int(segment.end//3600)}:{int(segment.end%3600//60):02}:{int(segment.end%60):02}.{int(segment.end%1*100):02}"
            
            # Word-level timing for karaoke effect
            line_parts = []
            for word in segment.words:
                duration_cs = int((word.end - word.start) * 100)  # Karaoke duration in centiseconds
                # The \k tag creates the progressive fill effect
                line_parts.append(f"{{\\k{duration_cs}}}{word.word.strip()}")
            line_text = " ".join(line_parts)
            
            # Write the dialogue line using the "Karaoke" style
            f.write(f"Dialogue: 0,{start_time},{end_time},Karaoke,,0,0,0,,{line_text}\n")

def get_audio_codec(format_str):
    """Maps common format names to FFmpeg codec names."""
    codec_map = {
        "mp3": "libmp3lame",
        "wav": "pcm_s16le",
        "flac": "flac"
    }
    return codec_map.get(format_str.lower(), "libmp3lame")

def process_media(
    source_path, output_dir_base, stem_volumes, pitch_shift, normalize_volume,
    speed_multiplier, generate_lyrics, whisper_model, karaoke_styles,
    cancel_flag, progress_callback, export_mode="Video", export_format="mp3",
    stems_to_export=None
):
    temp_processing_dir = None
    is_local_file = os.path.exists(source_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stems = ["vocals", "drums", "bass", "other"]
    if stems_to_export is None:
        stems_to_export = []

    progress_callback(f"Using processing device: {device}", 0)
    
    try:
        # Step 1: Setup
        check_cancel(cancel_flag); progress_callback("Step 1/8: Setting up...", 0)
        progress_callback("Getting media title...", 1)
        video_title = sanitize_filename(Path(source_path).stem) if is_local_file else \
            sanitize_filename(yt_dlp.YoutubeDL({'quiet': True, 'noplaylist': True, 'extract_flat': True})
                              .extract_info(source_path, download=False).get('title', 'video'))
        temp_processing_dir = os.path.join(output_dir_base, f"{video_title}_temp_{os.getpid()}")
        progress_callback(f"Creating temp directory: {temp_processing_dir}", 2)
        os.makedirs(temp_processing_dir, exist_ok=True)
        
        # Step 2: Acquire media
        check_cancel(cancel_flag); progress_callback("Step 2/8: Acquiring media...", 10)
        # ... (rest of media acquisition is unchanged, it is robust) ...
        if is_local_file:
            video_stream_file, full_audio_file = source_path, os.path.join(temp_processing_dir, 'full_audio.wav')
            progress_callback(f"Extracting audio from local file: {source_path}", 11)
            try:
                (ffmpeg
                 .input(source_path)
                 .output(full_audio_file, acodec='pcm_s16le', ar='44100', ac=2)
                 .run(overwrite_output=True, capture_stderr=True))
            except ffmpeg.Error as e:
                raise ProcessingError("FFmpeg failed during audio extraction.", e.stderr.decode('utf-8', errors='ignore'))
        else:
            ydl_opts = {'noplaylist': True, 'quiet': True, 'progress_hooks': [lambda d: progress_callback(f"Downloading: {d['_percent_str']}", 11)]}
            ydl_opts['outtmpl'] = os.path.join(temp_processing_dir, 'video_stream.%(ext)s')
            ydl_opts['format'] = 'bestvideo[ext=mp4]/best[ext=mp4]'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([source_path])
            video_stream_file = next(Path(temp_processing_dir).glob('video_stream.*'), None)
            if not video_stream_file: raise ProcessingError("Failed to download video stream.")

            ydl_opts['outtmpl'] = os.path.join(temp_processing_dir, 'full_audio.%(ext)s')
            ydl_opts['format'] = 'bestaudio/best'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([source_path])
            downloaded_audio_file = next(Path(temp_processing_dir).glob('full_audio.*'))
            full_audio_file = os.path.join(temp_processing_dir, 'full_audio.wav')
            if str(downloaded_audio_file) != full_audio_file:
                progress_callback("Converting downloaded audio to WAV...", 13)
                try:
                    (ffmpeg
                     .input(str(downloaded_audio_file))
                     .output(full_audio_file, acodec='pcm_s16le', ar='44100', ac=2)
                     .run(overwrite_output=True, capture_stderr=True))
                    os.remove(downloaded_audio_file)
                except ffmpeg.Error as e:
                    raise ProcessingError("FFmpeg failed during audio conversion.", e.stderr.decode('utf-8', errors='ignore'))
        progress_callback("Media acquisition complete.", 15)
        
        # Step 3: AI Lyrics
        subtitle_file = None
        if generate_lyrics and export_mode == "Video":
            check_cancel(cancel_flag); progress_callback(f"Step 3/8: Transcribing lyrics with '{whisper_model}' model...", 18)
            model = stable_whisper.load_model(whisper_model, device=device)
            result = model.transcribe(full_audio_file, fp16=torch.cuda.is_available())
            subtitle_file = os.path.join(temp_processing_dir, 'lyrics.ass')
            _generate_karaoke_subtitles(result, subtitle_file, karaoke_styles)
            progress_callback("Transcription complete.", 25)

        # ... (Step 4: Split Audio and Step 5: Demucs Separation are unchanged) ...
        check_cancel(cancel_flag); progress_callback("Step 4/8: Splitting audio...", 25)
        duration = float(ffmpeg.probe(full_audio_file)['format']['duration'])
        num_chunks = math.ceil(duration / CHUNK_DURATION_SECONDS) or 1
        chunks_dir = os.path.join(temp_processing_dir, 'chunks')
        os.makedirs(chunks_dir, exist_ok=True)
        for i in range(num_chunks):
            (ffmpeg.input(full_audio_file, ss=i * CHUNK_DURATION_SECONDS, t=CHUNK_DURATION_SECONDS)
             .output(os.path.join(chunks_dir, f'chunk_{i:03d}.wav')).run(overwrite_output=True, quiet=True))
        progress_callback("Splitting complete.", 30)

        check_cancel(cancel_flag); progress_callback(f"Step 5/8: Separating all stems...", 30)
        separated_dir = os.path.join(temp_processing_dir, 'separated')
        chunk_files = [str(f) for f in Path(chunks_dir).glob('*.wav')]
        # htdemucs is the default model name used by demucs output folders
        model_output_dir = os.path.join(separated_dir, "htdemucs") 
        command = [sys.executable, '-m', 'demucs', '--out', separated_dir, '--device', device] + chunk_files
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, universal_newlines=True, encoding='utf-8', errors='ignore')
        progress_regex = re.compile(r'(\d+)%\|')
        chunks_processed, last_percentage = 0, 0
        for line in iter(process.stderr.readline, ''):
            if match := progress_regex.search(line):
                percentage = int(match.group(1))
                if percentage < last_percentage and last_percentage > 95: chunks_processed = min(chunks_processed + 1, num_chunks)
                last_percentage = percentage
                overall_progress = 30 + ((chunks_processed + (percentage / 100)) / num_chunks) * 40
                progress_callback(f"AI Separation (Chunk {chunks_processed + 1}/{num_chunks}) - {percentage}%", overall_progress)
        process.wait()
        if process.returncode != 0: raise ProcessingError("Demucs failed.", process.stderr.read())
        progress_callback("AI separation complete.", 70)

        # ... (Step 6: Mix and Merge Stems is unchanged, it is robust) ...
        check_cancel(cancel_flag); progress_callback("Step 6/8: Mixing audio stems...", 75)
        mixed_chunks_dir = os.path.join(temp_processing_dir, 'mixed_chunks')
        os.makedirs(mixed_chunks_dir, exist_ok=True)
        for i in range(num_chunks):
            chunk_name = f'chunk_{i:03d}'
            stem_paths = {s: os.path.join(model_output_dir, chunk_name, f'{s}.wav') for s in stems}
            valid_stems = [s for s in stems if os.path.exists(stem_paths[s]) and stem_volumes.get(s, 0) > 0]
            if not valid_stems:
                (ffmpeg.input('anullsrc', format='lavfi', t=CHUNK_DURATION_SECONDS, r=44100)
                 .output(os.path.join(mixed_chunks_dir, f'mixed_{chunk_name}.wav')).run(overwrite_output=True, quiet=True))
                continue
            inputs_with_filter = [ffmpeg.input(stem_paths[s]).filter('volume', stem_volumes[s]) for s in valid_stems]
            mixed_output = ffmpeg.filter(inputs_with_filter, 'amix', inputs=len(valid_stems), duration='longest')
            mixed_output.output(os.path.join(mixed_chunks_dir, f'mixed_{chunk_name}.wav')).run(overwrite_output=True, quiet=True)
        
        progress_callback("Merging mixed chunks...", 80)
        mixed_audio_path = os.path.join(temp_processing_dir, 'mixed_audio.wav')
        mixed_chunk_files = sorted(Path(mixed_chunks_dir).glob('*.wav'))
        if not mixed_chunk_files: raise FileNotFoundError("No mixed audio chunks found for merging.")
        concat_list_file = os.path.join(temp_processing_dir, 'concat_list.txt')
        with open(concat_list_file, 'w', encoding='utf-8') as f:
            for file in mixed_chunk_files: f.write(f"file '{file.as_posix()}'\n")
        (ffmpeg.input(concat_list_file, format='concat', safe=0)
         .output(mixed_audio_path).run(overwrite_output=True, quiet=True))

        # Step 7: Apply Audio Effects (to the master mixed audio)
        final_audio_stream = ffmpeg.input(mixed_audio_path)
        if normalize_volume or pitch_shift != 0 or speed_multiplier != 1.0:
             check_cancel(cancel_flag); progress_callback("Step 7/8: Applying audio effects...", 85)
             if normalize_volume:
                 final_audio_stream = final_audio_stream.filter('loudnorm')
             if pitch_shift != 0:
                 pitch_multiplier = 2**(pitch_shift / 12.0)
                 final_audio_stream = final_audio_stream.filter('rubberband', pitch=pitch_multiplier)
             if speed_multiplier != 1.0:
                 final_audio_stream = final_audio_stream.filter('atempo', speed_multiplier)
        
        # --- EXPORT BRANCHING ---
        if export_mode == "Audio Only":
            final_audio_path = os.path.join(output_dir_base, f"{video_title}_Remixed.{export_format}")
            progress_callback(f"Exporting final audio to {final_audio_path}", 90)
            audio_codec = get_audio_codec(export_format)
            try:
                final_audio_stream.output(final_audio_path, acodec=audio_codec).run(overwrite_output=True, capture_stderr=True)
            except ffmpeg.Error as e:
                raise ProcessingError("FFmpeg failed while exporting final audio.", e.stderr.decode('utf-8', errors='ignore'))
            progress_callback(f"✅ Success! Audio saved to {final_audio_path}", 100)
            return

        if export_mode == "Stems Only":
            stem_out_dir = os.path.join(output_dir_base, f"{video_title}_stems")
            os.makedirs(stem_out_dir, exist_ok=True)
            audio_codec = get_audio_codec(export_format)
            for stem in stems_to_export:
                check_cancel(cancel_flag); progress_callback(f"Exporting stem: {stem}...", 90)
                stem_files = sorted(Path(model_output_dir).rglob(f"*/{stem}.wav"))
                if not stem_files: continue
                concat_list_file = os.path.join(temp_processing_dir, f"concat_{stem}.txt")
                with open(concat_list_file, 'w', encoding='utf-8') as f:
                    for file in stem_files: f.write(f"file '{file.as_posix()}'\n")
                stem_out_file = os.path.join(stem_out_dir, f"{video_title}_{stem}.{export_format}")
                try:
                    (ffmpeg.input(concat_list_file, format='concat', safe=0)
                     .output(stem_out_file, acodec=audio_codec).run(overwrite_output=True, capture_stderr=True))
                except ffmpeg.Error as e:
                    raise ProcessingError(f"FFmpeg failed while exporting {stem} stem.", e.stderr.decode('utf-8', errors='ignore'))
            progress_callback(f"✅ Success! Stems exported to {stem_out_dir}", 100)
            return

        # Step 8: Final Merge (Video Mode)
        progress_callback("Step 8/8: Merging final video...", 95)
        final_video_path = os.path.join(output_dir_base, f"{video_title}_Remixed.mp4")
        input_video = ffmpeg.input(str(video_stream_file))
        
        output_audio = final_audio_stream
        output_video = input_video['v']
        
        if speed_multiplier != 1.0:
            output_video = output_video.filter('setpts', f'{1.0/speed_multiplier}*PTS')
            
        if subtitle_file and os.path.exists(subtitle_file):
            progress_callback("Burning subtitles into video...", 97)
            output_video = output_video.filter('ass', filename=str(Path(subtitle_file).as_posix()))
        
        try:
            (ffmpeg
             .output(output_video, output_audio, final_video_path,
                     vcodec='libx264', pix_fmt='yuv420p', acodec='aac', audio_bitrate='320k', shortest=None)
             .run(overwrite_output=True, capture_stderr=True))
        except ffmpeg.Error as e:
            raise ProcessingError("FFmpeg failed during final video merge.", e.stderr.decode('utf-8', errors='ignore'))
        
        progress_callback(f"✅ Success! Final video saved to {final_video_path}", 100)

    except (ProcessingError, ffmpeg.Error, Exception) as e:
        error_message = f"❌ An error occurred: {e}"
        if isinstance(e, ProcessingError):
            error_message = f"❌ Processing Error: {e}\nDetails:\n{e.details}"
        elif isinstance(e, ffmpeg.Error):
             error_message = f"❌ FFmpeg Error:\n{e.stderr.decode('utf-8', errors='ignore')}"
        progress_callback(error_message, 100)
        raise e # Re-raise for the UI to catch
    finally:
        if temp_processing_dir and os.path.exists(temp_processing_dir):
            try:
                shutil.rmtree(temp_processing_dir)
                progress_callback("🧹 Cleanup complete.", 100)
            except OSError as e:
                progress_callback(f"🧹 Warning: Could not remove temp directory: {e}", 100)

# ---------- END: processing_logic.py (Fixed) ----------
