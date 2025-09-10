Of course. Here is a professional and detailed README file for your project. You can copy and paste this directly into the README.md file on your GitHub repository.

AI Media Processor Pro
A powerful desktop application for AI-powered audio processing, including a 4-stem mixer, karaoke video generation, and advanced audio effects. This tool allows users to process videos directly from YouTube or local files to create custom audio mixes, instrumental tracks, or full-featured karaoke videos with word-by-word synchronized lyrics.

(Note: You should replace this with a new screenshot of the final tabbed UI!)

Key Features
Flexible Inputs: Process videos and audio directly from YouTube URLs or local files on your computer. The application validates inputs to ensure they are valid paths or URLs.

AI Stem Remixer: Go beyond simple vocal removal with a full 4-stem audio mixer. Independently control the volume of vocals, drums, bass, and other instruments to create perfect instrumentals, acapellas, or custom-balanced tracks. This is powered by the Demucs library.

Automated Karaoke Video Generator:

AI-Powered Transcription: Automatically generates lyrics with highly accurate, word-level timestamps using selectable Whisper AI models (tiny, base, small, medium).


Synchronized Highlighting: Creates karaoke-style subtitle files (.ass format) where each word is highlighted exactly as it's sung.

Full Style Customization: Use the in-app controls to change the lyric font, size, and colors for both upcoming and highlighted text.


Smart Timing: Subtitles are configured to appear slightly before the words are sung for a seamless karaoke experience.

Advanced Audio Effects:


Pitch Shifting: Change the musical key of any song in semitones using the rubberband filter, perfect for vocal practice.


Playback Speed Control: Speed up or slow down the audio and video (from 0.5x to 2.0x) without changing the pitch, using the atempo filter.


Volume Normalization: Automatically adjust the final audio to a standard, consistent loudness for a professional sound using the loudnorm filter.

Professional Workflow:

Batch Processing: Add multiple jobs to a processing queue and run them all in order, unattended.

Dual Progress Bars: Get clear, real-time feedback on both the current task's progress and the overall queue completion.

Customizable UI: Switch between Light and Dark modes and choose from multiple color themes (including the custom "sweetkind" theme).

GPU Acceleration: The app automatically detects if a CUDA-enabled GPU is available and uses it for all AI tasks (Demucs and Whisper) to dramatically speed up processing. If no compatible GPU is found, it seamlessly falls back to using the CPU.

Tech Stack
GUI: CustomTkinter

Source Separation: Demucs

Transcription: stable-ts (Whisper)

Media I/O: FFmpeg (via ffmpeg-python)

Downloading: yt-dlp

AI Framework: PyTorch

Getting Started
Prerequisites
You must have a working installation of FFmpeg and a Conda environment (like Miniconda or Anaconda). For GPU acceleration, you need an NVIDIA graphics card with the appropriate CUDA drivers installed.

Installation
Clone the repository to your local machine.

Open an Anaconda/Miniconda Prompt and navigate to the project directory.

Create and activate the Conda environment:

Shell
# (One-time setup)
conda create --name remover_env python=3.9
conda activate remover_env

Install all required packages from the requirements.txt file:

Shell

pip install -r requirements.txt
If using a CUDA-enabled GPU, ensure you install the correct version of PyTorch by following the official instructions on their website.

Run the application:

Shell#
python app_ui.py  (in your newly created enviroment)

How to Use
Launch the application.

Set the Source (by pasting a URL or browsing for a file) and the Output Folder.

Configure your desired options in the Main Mixer, Audio Effects, and Karaoke tabs.

Click the "Add Job to Queue" button.

Repeat for any other videos you want to process.

Click the "▶️ Start Queue" button at the bottom to begin processing all jobs in the list.



License
This project is licensed under the GNU General Public License v3.0. This is required due to the licensing of core components like FFmpeg. This means if you distribute this software, you must also make the source code available.


