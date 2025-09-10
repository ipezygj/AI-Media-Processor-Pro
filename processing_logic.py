# File: processing_logic.py

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

# --- Custom Exception for Cancellation ---
class CancelledError(Exception):
    pass

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

def check_cancel(cancel_flag):
    if cancel_flag.is_set():
        raise CancelledError("Processing was cancelled by the user.")
        
def hex_to_ass_color(hex_color):
    """Converts a web hex color (#RRGGBB) to an ASS format color (&HBBGGRR)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6: return "&H00FFFFFF" # Default to white on error
    return f"&H{hex_color[4:6]}{hex_color[2:4]}{hex_color[0:2]}".upper()

def _generate_karaoke_subtitles(transcription_result, output_path, styles):
    """
    Generates an Advanced SubStation Alpha (.ass) subtitle file
    with user-defined, word-by-word karaoke highlighting.
    """
    upcoming_color = hex_to_ass_color(styles["upcoming_color"])
    highlight_color = hex_to_ass_color(styles["highlight_color"])
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\nTitle: Karaoke Lyrics\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n\n")
        f.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: Default,{styles['font_name']},{styles['font_size']},{upcoming_color},{upcoming_color},&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,2,2,10,10,20,1\n")
        f.write(f"Style: Highlight,{styles['font_name']},{styles['font_size']},{highlight_color},{highlight_color},&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,2,2,10,10,20,1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for segment in transcription_result.segments:
            start_s = max(0, segment.start - LYRIC_PRE_DISPLAY_OFFSET_SECONDS)
            start_time = f"{int(start_s//3600)}:{int(start_s%3600//60):02}:{int(start_s%60):02}.{int(start_s%1*100):02}"
            end_time = f"{int(segment.end//3600)}:{int(segment.end%3600//60):02}:{int(segment.end%60):02}.{int(segment.end%1*100):02}"
            line_text = "".join(f"{{\\K{int((word.end - word.start) * 100)}}}{word.word.strip()} " for word in segment.words)
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{{\\rHighlight}}{line_text.strip()}\n")

def process_media(source_path, output_dir_base, stem_volumes, pitch_shift, normalize_volume, speed_multiplier, generate_lyrics, whisper_model, karaoke_styles, cancel_flag, progress_callback):
    temp_processing_dir = None
    is_local_file = os.path.exists(source_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stems = ["vocals", "drums", "bass", "other"]

    try:
        # Steps 1-2
        check_cancel(cancel_flag); progress_callback("Step 1/8: Setting up...", 0)
        video_title = sanitize_filename(Path(source_path).stem) if is_local_file else \
            sanitize_filename(yt_dlp.YoutubeDL({'quiet': True, 'noplaylist': True}).extract_info(source_path, download=False).get('title', 'video'))
        temp_processing_dir = os.path.join(output_dir_base, f"{video_title}_temp")
        os.makedirs(temp_processing_dir, exist_ok=True)
        progress_callback(f"Outputting to: {output_dir_base}", 5)
        
        check_cancel(cancel_flag); progress_callback("Step 2/8: Acquiring media...", 10)
        if is_local_file:
            video_stream_file, full_audio_file = source_path, os.path.join(temp_processing_dir, 'full_audio.wav')
            ffmpeg.input(source_path).output(full_audio_file, acodec='pcm_s16le', ar='44100').run(overwrite_output=True, capture_stderr=True)
        else:
            ydl_opts = {'noplaylist': True, 'quiet': True, 'ratelimit': 5000000}
            ydl_opts['outtmpl'] = os.path.join(temp_processing_dir, 'video_stream.%(ext)s')
            ydl_opts['format'] = 'bestvideo[ext=mp4]/best[ext=mp4]'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([source_path])
            video_stream_file = next(Path(temp_processing_dir).glob('video_stream.*'))
            ydl_opts['outtmpl'] = os.path.join(temp_processing_dir, 'full_audio.%(ext)s')
            ydl_opts['format'] = 'bestaudio/best'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([source_path])
            downloaded_audio_file = next(Path(temp_processing_dir).glob('full_audio.*'))
            full_audio_file = os.path.join(temp_processing_dir, 'full_audio.wav')
            if str(downloaded_audio_file) != full_audio_file:
                ffmpeg.input(str(downloaded_audio_file)).output(full_audio_file, acodec='pcm_s16le', ar='44100').run(overwrite_output=True, capture_stderr=True)
                os.remove(downloaded_audio_file)
        progress_callback("Media acquisition complete.", 15)
        
        # Step 2.5: AI Lyrics
        subtitle_file = None
        if generate_lyrics:
            check_cancel(cancel_flag); progress_callback(f"Step 3/8: Transcribing lyrics...", 18)
            model = stable_whisper.load_model(whisper_model, device=device)
            result = model.transcribe(full_audio_file, fp16=torch.cuda.is_available())
            subtitle_file = os.path.join(temp_processing_dir, 'lyrics.ass')
            _generate_karaoke_subtitles(result, subtitle_file, karaoke_styles)
            progress_callback("Transcription complete.", 25)

        # Step 3: Split Audio
        check_cancel(cancel_flag); progress_callback("Step 4/8: Splitting audio...", 25)
        duration = float(ffmpeg.probe(full_audio_file)['format']['duration'])
        num_chunks = math.ceil(duration / CHUNK_DURATION_SECONDS) or 1
        chunks_dir = os.path.join(temp_processing_dir, 'chunks')
        os.makedirs(chunks_dir, exist_ok=True)
        for i in range(num_chunks):
            ffmpeg.input(full_audio_file, ss=i * CHUNK_DURATION_SECONDS, t=CHUNK_DURATION_SECONDS).output(os.path.join(chunks_dir, f'chunk_{i:03d}.wav')).run(overwrite_output=True, capture_stderr=True)
        progress_callback("Splitting complete.", 30)

        # Step 4: Demucs Separation
        check_cancel(cancel_flag); progress_callback(f"Step 5/8: Separating all stems...", 30)
        separated_dir = os.path.join(temp_processing_dir, 'separated')
        chunk_files = [str(f) for f in Path(chunks_dir).glob('*.wav')]
        command = [sys.executable, '-m', 'demucs', '--out', separated_dir, '--device', device] + chunk_files
        process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True, universal_newlines=True, encoding='utf-8', errors='ignore')
        progress_regex = re.compile(r'(\d+)\%\|')
        chunks_processed, last_percentage = 0, 0
        for line in iter(process.stderr.readline, ''):
            if match := progress_regex.search(line):
                percentage = int(match.group(1))
                if percentage < last_percentage and last_percentage > 95: chunks_processed = min(chunks_processed + 1, num_chunks - 1)
                last_percentage = percentage
                overall_progress = 30 + ((chunks_processed + (percentage / 100)) / num_chunks) * 40
                progress_callback(f"Step 5/8: AI Separation (Chunk {chunks_processed + 1}/{num_chunks}) - {percentage}%", overall_progress)
        process.wait()
        if process.returncode != 0: raise Exception("Demucs failed.")
        progress_callback("AI separation complete.", 70)

        # Step 5: Mix and Merge Stems
        check_cancel(cancel_flag); progress_callback("Step 6/8: Mixing audio stems...", 75)
        mixed_chunks_dir = os.path.join(temp_processing_dir, 'mixed_chunks')
        os.makedirs(mixed_chunks_dir, exist_ok=True)
        model_name = "htdemucs"
        for i in range(num_chunks):
            chunk_name = f'chunk_{i:03d}'
            stem_paths = {s: os.path.join(separated_dir, model_name, chunk_name, f'{s}.wav') for s in stems}
            valid_stems = [s for s in stems if os.path.exists(stem_paths[s]) and stem_volumes.get(s, 0) > 0]
            if not valid_stems:
                ffmpeg.input('anullsrc', format='lavfi', t=CHUNK_DURATION_SECONDS).output(os.path.join(mixed_chunks_dir, f'mixed_{chunk_name}.wav')).run(overwrite_output=True, capture_stderr=True)
                continue
            inputs_with_filter = [ffmpeg.input(stem_paths[s]).filter('volume', stem_volumes[s]) for s in valid_stems]
            mixed_output = ffmpeg.filter(inputs_with_filter, 'amix', inputs=len(valid_stems), duration='longest')
            mixed_output.output(os.path.join(mixed_chunks_dir, f'mixed_{chunk_name}.wav')).run(overwrite_output=True, capture_stderr=True)
        
        progress_callback("Merging mixed chunks...", 80)
        mixed_audio_path = os.path.join(temp_processing_dir, 'mixed_audio.wav')
        mixed_chunk_files = sorted(Path(mixed_chunks_dir).glob('*.wav'))
        if not mixed_chunk_files: raise FileNotFoundError("No mixed audio chunks found.")
        concat_list_file = os.path.join(temp_processing_dir, 'concat_list.txt')
        with open(concat_list_file, 'w', encoding='utf-8') as f:
            for file in mixed_chunk_files: f.write(f"file '{file.as_posix()}'\n")
        ffmpeg.input(concat_list_file, format='concat', safe=0).output(mixed_audio_path).run(overwrite_output=True, capture_stderr=True)

        # Step 6: Apply Advanced Audio Effects
        final_audio_stream = ffmpeg.input(mixed_audio_path)
        if normalize_volume or pitch_shift != 0:
            progress_callback("Step 7/8: Applying audio effects...", 85)
            if normalize_volume:
                final_audio_stream = final_audio_stream.filter('loudnorm')
            if pitch_shift != 0:
                pitch_multiplier = 2**(pitch_shift / 12.0)
                final_audio_stream = final_audio_stream.filter('rubberband', pitch=pitch_multiplier)

        # Step 8: Final Merge
        progress_callback("Step 8/8: Merging final video...", 95)
        final_video_path = os.path.join(output_dir_base, f"{video_title}_Remixed.mp4")
        input_video = ffmpeg.input(str(video_stream_file))
        
        output_audio_stream = final_audio_stream
        output_video_stream = input_video['v']
        
        if speed_multiplier != 1.0:
            output_audio_stream = output_audio_stream.filter('atempo', speed_multiplier)
            output_video_stream = input_video.filter('setpts', f'{1.0/speed_multiplier}*PTS')
            
        if subtitle_file and os.path.exists(subtitle_file):
            progress_callback("Burning subtitles into video...", 97)
            # FIXED: Added filename= to handle special characters in path
            output_video_stream = output_video_stream.filter('ass', filename=subtitle_file)
        
        ffmpeg.output(
            output_video_stream, 
            output_audio_stream, 
            final_video_path, 
            vcodec='libx264', 
            acodec='aac',
            audio_bitrate='320k',
            shortest=None
        ).run(overwrite_output=True, capture_stderr=True)
        
        progress_callback(f"✅ Success! Final video saved to {final_video_path}", 100)

    except ffmpeg.Error as e:
        error_details = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "No details from FFmpeg."
        progress_callback(f"❌ FFmpeg Error:\n{error_details}", 100)
        raise e
    except Exception as e:
        progress_callback(f"❌ An unexpected error occurred: {e}", 100)
        raise e
    finally:
        if temp_processing_dir and os.path.exists(temp_processing_dir):
            shutil.rmtree(temp_processing_dir, ignore_errors=True)
            progress_callback("🧹 Cleanup complete.", 100)