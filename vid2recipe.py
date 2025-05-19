

import os
import cv2
import torch
import tempfile
import shutil
import time
import logging
import subprocess
from flask import Flask, request, render_template_string, flash, redirect, url_for, make_response, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from urllib.parse import urlparse
import yt_dlp
from dotenv import load_dotenv
from openai import OpenAI
import re
import numpy as np
import webvtt
import pysrt
import uuid

# === CONFIG ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
device = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_FOLDER = "Uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
DEBUG_MODE = True  # Set to True to keep files for debugging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === LOAD BLIP MODEL ===
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# === FLASK APP ===
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB limit
app.secret_key = os.urandom(24)  # Required for flash messages

# === HTML TEMPLATE ===
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Video to Recipe</title>
    <style>
        .loading-container { display: none; text-align: center; margin-top: 20px; }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        #status-message { margin-top: 10px; font-weight: bold; }
        .error { color: red; }
        .warning { color: orange; }
        .info { color: blue; }
        pre { background: #f4f4f4; padding: 10px; white-space: pre-wrap; }
        ul { list-style-type: disc; margin-left: 20px; }
    </style>
</head>
<body>
    <h1>Upload Cooking Video or Share Link</h1>
    <div id="form-container">
        <label>Select a video file:</label><br>
        <input type="file" id="video" accept="video/*"><br><br>

        <label>Or enter a video URL (e.g., YouTube):</label><br>
        <input type="text" id="video_url" placeholder="https://www.youtube.com/watch?v=example" size="50"><br><br>

        <label>Dish name (optional, e.g., Biryani):</label><br>
        <input type="text" id="dish_name" placeholder="Enter dish name" size="50"><br><br>

        <label>Dietary Preference (optional, e.g., Vegan, Gluten-Free):</label><br>
        <input type="text" id="dietary_preference" placeholder="Enter dietary preference" size="50"><br><br>

        <button onclick="submitForm()">Generate Recipe</button>
    </div>

    <div id="loading" class="loading-container">
        <div class="spinner"></div>
        <p id="status-message">Processing...</p>
    </div>

    <div id="results">
        {% if not request.form.get('dish_name') %}
          <p class="warning">Warning: No dish name provided. The recipe is based on an inferred dish, which may affect accuracy. Consider specifying a dish name for better results.</p>
        {% endif %}

        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            {% for category, message in messages %}
              {% if category == 'video_title' %}
                <p class="info">Video Title: {{ message }}</p>
              {% elif category == 'inferred_dish' %}
                <p class="info">Inferred Dish: {{ message }}</p>
              {% elif category == 'captions' %}
                <h2>Visual Captions</h2>
                <p>Note: These captions are generated from video frames. If they seem inaccurate, the recipe may be less reliable.</p>
                <ul>
                  {% for cap in message %}
                    <li>{{ cap }}</li>
                  {% endfor %}
                </ul>
              {% elif category == 'recipe' %}
                <h2>Original Recipe</h2>
                <pre>{{ message }}</pre>
              {% elif category == 'dietary_recipe' %}
                <h2>Dietary Preference Recipe</h2>
                <pre>{{ message }}</pre>
              {% elif category == 'warning' %}
                <p class="warning">Warning: {{ message }}</p>
              {% elif category == 'error' %}
                <p class="error">Error: {{ message }}</p>
              {% endif %}
            {% endfor %}
          {% endif %}
        {% endwith %}
    </div>

    <script>
        function showLoading(status) {
            document.getElementById('form-container').style.display = 'none';
            document.getElementById('loading').style.display = 'block';
            document.getElementById('status-message').innerText = status;
        }

        function hideLoading() {
            document.getElementById('form-container').style.display = 'block';
            document.getElementById('loading').style.display = 'none';
        }

        async function submitForm() {
            const video = document.getElementById('video').files[0];
            const videoUrl = document.getElementById('video_url').value.trim();
            const dishName = document.getElementById('dish_name').value.trim();
            const dietaryPreference = document.getElementById('dietary_preference').value.trim();

            if (!video && !videoUrl) {
                alert('Please upload a video or provide a valid video URL.');
                return;
            }

            const formData = new FormData();
            if (video) {
                formData.append('video', video);
            }
            formData.append('video_url', videoUrl);
            formData.append('dish_name', dishName);
            formData.append('dietary_preference', dietaryPreference);

            const statusMessages = [
                'Downloading video...',
                'Extracting audio...',
                'Transcribing audio...',
                'Extracting frames...',
                'Generating captions...',
                'Generating recipe...'
            ];
            let statusIndex = 0;

            showLoading(statusMessages[statusIndex]);
            const statusInterval = setInterval(() => {
                statusIndex = (statusIndex + 1) % statusMessages.length;
                document.getElementById('status-message').innerText = statusMessages[statusIndex];
            }, 3000);

            try {
                const response = await fetch('/', {
                    method: 'POST',
                    body: formData
                });
                clearInterval(statusInterval);
                hideLoading();
                if (response.ok) {
                    const html = await response.text();
                    // Update the results div with the new content
                    document.getElementById('results').innerHTML = html.match(/<div id="results">([\s\S]*?)<\/div>/)[1];
                    // Ensure the form is visible again
                    document.getElementById('form-container').style.display = 'block';
                } else {
                    const error = await response.text();
                    alert('Error: ' + error);
                }
            } catch (error) {
                clearInterval(statusInterval);
                hideLoading();
                alert('Error: ' + error.message);
            }
        }
    </script>
</body>
</html>
"""

# === UTIL: Check FFmpeg ===
def check_ffmpeg():
    """Check if FFmpeg is installed."""
    return shutil.which("ffmpeg") is not None

# === UTIL: Check Folder Permissions ===
def check_folder_permissions(folder):
    """Check if the folder is writable."""
    try:
        test_file = os.path.join(folder, f"test_{int(time.time())}.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception as e:
        logger.error(f"Folder {folder} is not writable: {str(e)}")
        return False

# === UTIL: Extract Audio ===
def extract_audio(video_path, audio_path):
    """Extract audio from video using FFmpeg."""
    if not check_ffmpeg():
        raise Exception("FFmpeg is not installed. Cannot extract audio.")
    
    logger.info(f"Extracting audio from {video_path} to {audio_path}")
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", "-y", audio_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Audio extraction successful: {result.stdout}")
        return audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio extraction failed: {e.stderr}")
        raise Exception(f"Failed to extract audio: {e.stderr}")

# === UTIL: Process Subtitles ===
def process_subtitles(subtitle_path):
    """Parse VTT or SRT subtitle file and extract text."""
    if not os.path.exists(subtitle_path):
        return "", "No subtitles found."
    
    try:
        subtitle_text = []
        if subtitle_path.endswith('.vtt'):
            for caption in webvtt.read(subtitle_path):
                clean_text = re.sub(r'<.*?>', '', caption.text).strip()  # Remove HTML tags
                if clean_text:
                    subtitle_text.append(clean_text)
        elif subtitle_path.endswith('.srt'):
            subs = pysrt.open(subtitle_path)
            for sub in subs:
                clean_text = re.sub(r'<.*?>', '', sub.text).strip()
                if clean_text:
                    subtitle_text.append(clean_text)
        result = " ".join(subtitle_text)
        if not result:
            return "", "Subtitles are empty or unreadable."
        logger.info(f"Subtitles processed: {result[:200]}...")
        return result, ""
    except Exception as e:
        logger.error(f"Subtitle processing error: {str(e)}")
        return "", f"Failed to process subtitles: {str(e)}"

# === UTIL: Transcribe Audio ===
def transcribe_audio(audio_path, subtitle_paths=None):
    """Use subtitles if available, otherwise transcribe audio with Whisper."""
    subtitle_paths = subtitle_paths or []
    
    # Try each subtitle file, prioritizing English
    for subtitle_path in subtitle_paths:
        if subtitle_path and os.path.exists(subtitle_path):
            subtitle_text, subtitle_warning = process_subtitles(subtitle_path)
            if subtitle_text:
                logger.info(f"Using subtitles from {subtitle_path}")
                return subtitle_text, subtitle_warning
            if subtitle_warning:
                logger.warning(subtitle_warning)
    
    # Fallback to Whisper transcription
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        logger.info(f"Audio transcription: {transcription[:200]}...")
        # Check for cooking relevance
        cooking_terms = ["cook", "bake", "fry", "boil", "ingredient", "recipe", "mix", "stir", "chop", "slice", "biryani", "curry", "spice", "rice", "saffron"]
        word_count = len(transcription.split())
        cooking_term_count = sum(1 for term in cooking_terms if term in transcription.lower())
        cooking_term_ratio = cooking_term_count / word_count if word_count > 0 else 0
        if word_count > 50 and cooking_term_ratio < 0.05:
            logger.warning("Transcription appears to be a song or non-cooking audio.")
            return "", "Audio seems unrelated to cooking (possibly a song). Relying on visual captions."
        return transcription, ""
    except Exception as e:
        logger.error(f"Whisper transcription error: {str(e)}")
        return "", f"Failed to transcribe audio: {str(e)}"

# === UTIL: Re-encode Video ===
def reencode_video(input_path, output_path):
    """Re-encode video to H.264 MP4 using FFmpeg."""
    if not check_ffmpeg():
        raise Exception("FFmpeg is not installed. Cannot re-encode video.")
    
    logger.info(f"Re-encoding video from {input_path} to {output_path}")
    cmd = [
        "ffmpeg", "-i", input_path, "-c:v", "libx264", "-c:a", "aac",
        "-preset", "fast", "-crf", "23", "-y", output_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Re-encoding successful: {result.stdout}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg re-encoding failed: {e.stderr}")
        raise Exception(f"Failed to re-encode video: {e.stderr}")

# === UTIL: Download Video with yt-dlp ===
def download_video_from_url(video_url):
    """Download video, subtitles, and extract title with unique filename using yt-dlp."""
    timestamp = int(time.time() * 1000)
    output_path = os.path.join(UPLOAD_FOLDER, f"video_{timestamp}.%(ext)s")
    video_path = os.path.join(UPLOAD_FOLDER, f"video_{timestamp}.mp4")
    subtitle_path_base = os.path.join(UPLOAD_FOLDER, f"subtitles_{timestamp}")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4][vcodec^=avc1]/best',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en', 'all'],
        'subtitlesformat': 'vtt,srt',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        } if check_ffmpeg() else None],
        'quiet': False,
        'noplaylist': True,
        'http_headers': {'User-Agent': 'Mozilla/5.0'},
        'retries': 3,
        'no_cache_dir': True,
    }

    logger.info(f"Downloading video, subtitles, and title from {video_url} to {output_path}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_title = info.get('title', 'Unknown Title')
            logger.info(f"Extracted video title: {video_title}")

        if os.path.exists(video_path):
            final_path = video_path
        else:
            for ext in ['.mkv', '.webm']:
                alt_path = os.path.join(UPLOAD_FOLDER, f"video_{timestamp}{ext}")
                if os.path.exists(alt_path):
                    final_path = reencode_video(alt_path, video_path)
                    os.remove(alt_path)
                    break
            else:
                raise Exception("Failed to create video file. Check if the URL is valid or download completed.")

        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        logger.info(f"Video file created: {final_path} ({size_mb:.2f} MB)")

        if size_mb > 100:
            raise Exception(f"Video too large ({size_mb:.2f} MB)")

        subtitle_paths = []
        for ext in ['.en.vtt', '.vtt', '.en.srt', '.srt']:
            sub_path = f"{subtitle_path_base}{ext}"
            if os.path.exists(sub_path):
                subtitle_paths.append(sub_path)
                logger.info(f"Found subtitle file: {sub_path}")
        
        return final_path, subtitle_paths, video_title

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"Download error: {str(e)}")
        if "ffmpeg" in str(e).lower() and check_ffmpeg():
            raise Exception("FFmpeg error: Could not convert video.")
        raise Exception(f"Download failed: {str(e)}")

# === UTIL: Validate Video ===
def validate_video(video_path):
    """Verify if the video file is readable by OpenCV."""
    if not os.path.exists(video_path):
        return False, f"Video file does not exist: {video_path}"
    
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        vidcap.release()
        return False, "Cannot open video file (possibly corrupt, unsupported format, or missing codec like AV1)."
    
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    
    vidcap.release()
    
    if frame_count <= 0:
        return False, "Video has no readable frames."
    if duration < 1:
        return False, f"Video too short ({duration:.2f} seconds)."
    
    return True, f"Valid video: {frame_count} frames, {duration:.2f} seconds, {fps:.2f} FPS."

# === UTIL: Extract Frames ===
def extract_frames(video_path, max_frames=25):
    """Extract key frames based on frame differences with debugging support."""
    logger.info(f"Extracting frames from {video_path}")
    
    is_valid, validation_message = validate_video(video_path)
    if not is_valid:
        logger.error(f"Video validation failed: {validation_message}")
        if "unsupported format" in validation_message and check_ffmpeg():
            try:
                timestamp = int(time.time() * 1000)
                reencoded_path = os.path.join(UPLOAD_FOLDER, f"reencoded_{timestamp}.mp4")
                reencode_video(video_path, reencoded_path)
                logger.info(f"Re-encoded video to {reencoded_path}")
                video_path = reencoded_path
                is_valid, validation_message = validate_video(video_path)
                if not is_valid:
                    raise Exception(validation_message)
            except Exception as e:
                raise Exception(f"Re-encoding failed: {str(e)}")
        else:
            raise Exception(validation_message)
    
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    logger.info(f"Video duration: {duration:.2f} seconds, max_frames: {max_frames}")
    
    saved_frames = []
    prev_frame = None
    count = 0
    success, frame = vidcap.read()
    
    debug_folder = os.path.join(UPLOAD_FOLDER, "debug_frames")
    if DEBUG_MODE:
        os.makedirs(debug_folder, exist_ok=True)
    
    while success and len(saved_frames) < max_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            diff_score = float(np.mean(diff))
            if diff_score > 30:
                try:
                    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
                    h, w = frame.shape[:2]
                    crop_size = int(min(h, w) * 0.9)
                    start_x = (w - crop_size) // 2
                    start_y = (h - crop_size) // 2
                    frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    saved_frames.append(pil_img)
                    logger.info(f"Extracted key frame {len(saved_frames)} at {count/fps:.2f} seconds (diff: {diff_score:.2f})")
                    if DEBUG_MODE:
                        pil_img.save(os.path.join(debug_folder, f"frame_{len(saved_frames)}_{count}.jpg"))
                except Exception as e:
                    logger.error(f"Failed to process frame at count {count}: {str(e)}")
        
        prev_frame = gray
        success, frame = vidcap.read()
        count += 1
        if fps > 0:
            skip_frames = int(fps * 1.0)
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, count + skip_frames)
            count += skip_frames
            success, frame = vidcap.read()
    
    vidcap.release()
    
    if len(saved_frames) < 5 and duration > 1:
        vidcap = cv2.VideoCapture(video_path)
        interval = max(1, int(fps * (duration / max_frames)))
        success, frame = vidcap.read()
        count = 0
        while success and len(saved_frames) < max_frames:
            if count % interval == 0:
                try:
                    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=5)
                    h, w = frame.shape[:2]
                    crop_size = int(min(h, w) * 0.9)
                    start_x = (w - crop_size) // 2
                    start_y = (h - crop_size) // 2
                    frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    saved_frames.append(pil_img)
                    logger.info(f"Extracted fallback frame {len(saved_frames)} at {count/fps:.2f} seconds")
                    if DEBUG_MODE:
                        pil_img.save(os.path.join(debug_folder, f"fallback_frame_{len(saved_frames)}_{count}.jpg"))
                except Exception as e:
                    logger.error(f"Failed to process fallback frame at count {count}: {str(e)}")
            success, frame = vidcap.read()
            count += 1
        vidcap.release()
    
    if not saved_frames:
        raise Exception("No frames extracted. Possible issues: video too short, unsupported format, or corrupted file.")
    
    logger.info(f"Extracted {len(saved_frames)} frames from {video_path}")
    return saved_frames

# === UTIL: Caption One Image ===
def caption_image(pil_image):
    inputs = processor(pil_image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    logger.info(f"Generated caption: {caption}")
    return caption

# === UTIL: Validate Captions ===
def validate_captions(captions):
    """Remove outlier captions based on keyword overlap."""
    if len(captions) <= 3:
        return captions
    
    cooking_keywords = [
        "food", "ingredient", "cook", "chop", "stir", "fry", "bake", "mix", "boil", "grill",
        "vegetable", "meat", "spice", "rice", "saffron", "curry", "pan", "pot", "onion", "garlic",
        "turmeric", "ginger", "sauce", "dough", "fruit", "cheese", "noodles", "lentil", "bread",
        "miso", "kimchi", "lemongrass", "soy", "tofu", "sesame", "chili", "coconut", "basil",
        "nori", "gochujang", "galangal", "fish sauce", "wasabi", "mirin", "sake"
    ]
    
    def keyword_overlap(cap):
        cap_words = set(cap.lower().split())
        return sum(1 for word in cooking_keywords if word in cap_words)
    
    valid_captions = [cap for cap in captions if keyword_overlap(cap) >= 1]
    
    if not valid_captions:
        valid_captions = captions
        logger.warning("No captions passed overlap validation. Using all captions.")
    else:
        logger.info(f"Validated captions: {valid_captions}")
    
    return valid_captions

# === UTIL: Filter Captions ===
def filter_captions(captions, inferred_cuisine=""):
    """Filter and rank captions by relevance to cooking with cuisine context."""
    vague_terms = ["person", "standing", "room", "table", "background", "image", "scene", "object"]
    cooking_terms = [
        "food", "ingredient", "cook", "chop", "stir", "fry", "bake", "mix", "boil", "grill",
        "vegetable", "meat", "spice", "rice", "saffron", "curry", "pan", "pot", "sauce", "dough"
    ]
    specific_terms = [
        "saffron", "basmati", "turmeric", "ginger", "garlic", "onion", "chicken", "beef", "tofu",
        "noodles", "bread", "cheese", "curry", "rice", "lentil", "banana", "mango", "fruit",
        "miso", "kimchi", "lemongrass", "soy", "sesame", "chili", "coconut", "basil",
        "nori", "gochujang", "galangal", "fish sauce", "wasabi", "mirin", "sake"
    ]
    cooking_actions = ["cook", "chop", "stir", "fry", "bake", "mix", "boil", "grill"]
    
    cuisine_keywords = {
        "indian": ["saffron", "turmeric", "curry", "biryani", "lentil", "basmati", "ginger"],
        "french": ["butter", "cream", "wine", "baguette", "sauce", "herbs"],
        "english": ["beef", "potato", "pudding", "pie", "fish", "mash"],
        "chinese": ["soy", "noodles", "wok", "sesame", "ginger", "oyster sauce"],
        "japanese": ["miso", "sushi", "rice", "soy", "tofu", "nori", "wasabi", "mirin"],
        "korean": ["kimchi", "gochujang", "sesame", "beef", "rice", "banchan"],
        "thai": ["lemongrass", "coconut", "chili", "basil", "curry", "galangal", "fish sauce"]
    }
    
    def score_caption(cap):
        score = 0
        cap_lower = cap.lower()
        if len(cap.split()) > 3:
            score += 2
        score += sum(3 for term in specific_terms if term in cap_lower)
        score += sum(1 for term in cooking_terms if term in cap_lower)
        score -= sum(1 for term in vague_terms if term in cap_lower)
        if inferred_cuisine in cuisine_keywords:
            score += sum(2 for term in cuisine_keywords[inferred_cuisine] if term in cap_lower)
        if any(fruit in cap_lower for fruit in ["banana", "mango", "fruit"]):
            if not any(action in cap_lower for action in cooking_actions):
                score -= 2
        return score

    scored_captions = [(cap, score_caption(cap)) for cap in captions]
    filtered = [cap for cap, score in sorted(scored_captions, key=lambda x: x[1], reverse=True) if score > 0]
    
    if not filtered:
        filtered = [cap for cap, _ in scored_captions if len(cap.split()) > 3]
        if not filtered:
            filtered = captions
            logger.warning("No relevant captions found. Using all captions, which may affect recipe accuracy.")
    
    filtered = validate_captions(filtered)
    
    overlap_scores = [sum(1 for word in cooking_terms if word in cap.lower()) for cap in filtered]
    if filtered and max(overlap_scores) - min(overlap_scores) > 5:
        logger.warning("Captions may be inconsistent, which could affect recipe accuracy.")
        flash("Warning: Captions vary widely, which may affect recipe accuracy.", "warning")
    
    logger.info(f"Filtered captions: {filtered}")
    return filtered[:10]

# === UTIL: Suggest Dish Name ===
def suggest_dish_name(captions, audio_transcription, video_title):
    """Suggest a dish name and cuisine based on captions, audio transcription, and video title."""
    keywords = []
    for cap in captions:
        cap_lower = cap.lower()
        keywords.extend([word for word in cap_lower.split() if word in [
            "rice", "saffron", "biryani", "curry", "noodles", "stir-fry", "burger", "steak", "pasta",
            "sushi", "tofu", "lentil", "turmeric", "ginger", "miso", "kimchi", "lemongrass", "soy",
            "nori", "gochujang", "galangal", "wasabi", "mirin", "sake"
        ]])
    if audio_transcription:
        keywords.extend([word for word in audio_transcription.lower().split() if word in [
            "biryani", "curry", "burger", "steak", "pasta", "sushi", "stir-fry", "lentil", "rice",
            "spice", "miso", "kimchi", "lemongrass", "soy", "nori", "gochujang", "galangal", "wasabi"
        ]])
    if video_title:
        keywords.extend([word for word in video_title.lower().split() if word in [
            "biryani", "curry", "burger", "steak", "pasta", "sushi", "stir-fry", "lentil", "rice",
            "spice", "miso", "kimchi", "lemongrass", "soy", "nori", "gochujang", "galangal", "wasabi"
        ]])
    
    dish_map = {
        "biryani": (["rice", "saffron", "curry", "biryani", "turmeric", "ginger"], "indian"),
        "curry": (["curry", "spice", "vegetable", "meat", "lentil", "turmeric"], "indian"),
        "butter chicken": (["chicken", "butter", "curry", "spice"], "indian"),
        "burger": (["burger", "bun", "patty"], "english"),
        "stir-fry": (["noodles", "stir-fry", "vegetable", "tofu", "soy"], "chinese"),
        "pasta": (["pasta", "noodles", "sauce"], "italian"),
        "lentil soup": (["lentil", "spice", "vegetable"], "indian"),
        "sushi": (["sushi", "rice", "fish", "nori"], "japanese"),
        "ramen": (["noodles", "miso", "soy", "broth"], "japanese"),
        "bibimbap": (["rice", "kimchi", "sesame", "vegetable"], "korean"),
        "pad thai": (["noodles", "lemongrass", "chili", "peanut"], "thai"),
        "croissant": (["butter", "dough", "bake"], "french")
    }
    
    dish_scores = {dish: sum(1 for kw in keywords if kw in terms) for dish, (terms, _) in dish_map.items()}
    if dish_scores:
        suggested_dish = max(dish_scores, key=dish_scores.get)
        if dish_scores[suggested_dish] > 1:
            cuisine = dish_map[suggested_dish][1]
            logger.info(f"Suggested dish name: {suggested_dish}, cuisine: {cuisine}")
            return suggested_dish, cuisine
    return "", ""

# === UTIL: Check Caption Coherence ===
def check_caption_coherence(captions):
    """Check if captions describe a coherent cooking process."""
    cooking_actions = ["chop", "stir", "fry", "bake", "mix", "boil", "grill"]
    ingredients = [
        "rice", "meat", "vegetable", "spice", "onion", "garlic", "saffron", "lentil", "noodles",
        "miso", "kimchi", "lemongrass", "soy", "tofu", "fruit", "nori", "gochujang", "galangal"
    ]
    
    action_count = sum(1 for cap in captions if any(action in cap.lower() for action in cooking_actions))
    ingredient_count = sum(1 for cap in captions if any(ing in cap.lower() for ing in ingredients))
    
    if action_count < 2 or ingredient_count < 2:
        logger.warning("Captions may not describe a coherent cooking process.")
        return False, "Captions are too vague or unrelated. Recipe accuracy may be affected."
    return True, ""

# === UTIL: Generate Recipe from Captions via GPT ===
def generate_recipe_from_captions(captions, audio_transcription="", dish_name="", dietary_preference="", inferred_cuisine="", video_title=""):
    """Generate complete original and dietary-preference recipes with cuisine-specific guidance."""
    dish_context = f" for {dish_name}" if dish_name else " (no specific dish name provided, infer the dish based on visual, audio, and video title data)"
    dietary_context = f" meeting the dietary requirements of {dietary_preference} (e.g., replacing non-compliant ingredients with suitable alternatives like tofu for meat in vegan recipes)" if dietary_preference else ""
    audio_context = f"\n\nAudio transcription or subtitles from the video:\n{audio_transcription}" if audio_transcription else ""
    title_context = f"\n\nVideo title: {video_title}" if video_title else ""
    
    cuisine_guidance = {
        "indian": "Use spices like saffron, turmeric, or cumin, and techniques like layering for biryani or simmering for curry. Prefer basmati rice and traditional cookware like a handi.",
        "french": "Incorporate butter, cream, or wine-based sauces, and techniques like braising or baking. Use fresh herbs and precise measurements.",
        "english": "Focus on hearty ingredients like beef, potatoes, or fish, with methods like roasting or frying. Keep recipes simple and traditional.",
        "chinese": "Use soy sauce, sesame oil, or ginger, and techniques like stir-frying in a wok. Emphasize balance of flavors.",
        "japanese": "Incorporate miso, soy, or rice, with techniques like steaming or raw preparation for sushi. Use minimal seasoning and ingredients like nori or wasabi.",
        "korean": "Use kimchi, gochujang, or sesame, with techniques like grilling or fermenting. Emphasize bold flavors and side dishes (banchan).",
        "thai": "Use lemongrass, coconut milk, or chili, with techniques like curry-making or stir-frying. Balance sweet, sour, and spicy with ingredients like galangal or fish sauce."
    }
    cuisine_prompt = cuisine_guidance.get(inferred_cuisine, "Adapt to the most likely cuisine based on ingredients and methods described.") if inferred_cuisine else "Adapt to the most likely cuisine based on ingredients and methods described."
    
    prompt = (
        f"You are an expert chef specializing in global cuisines (Indian, French, English, Chinese, Japanese, Korean, Thai, etc.). "
        f"Based on the following visual descriptions from a cooking video{dish_context}, "
        f"the video title, and audio transcription, generate two detailed recipes, each clearly separated by '###'. "
        f"Ensure both recipes are complete, including:\n"
        f"- A title (e.g., 'Traditional Biryani Recipe').\n"
        f"- An 'Ingredients' section with precise measurements in metric units (e.g., '200g basmati rice').\n"
        f"- An 'Instructions' section with numbered, step-by-step cooking steps.\n"
        f"- A 'Tips' section for authenticity (e.g., cookware, ingredient quality).\n"
        f"Recipes:\n"
        f"1. An authentic recipe reflecting traditional techniques of the cuisine. {cuisine_prompt} Include only ingredients and actions explicitly mentioned or strongly implied by the captions, subtitles, audio, or video title. "
        f"If no dish name is provided, use the video title to guide dish inference (e.g., 'Chicken Biryani' from title suggests biryani). "
        f"If data is vague, generate a recipe matching the described ingredients and methods, prioritizing the inferred cuisine. "
        f"Clearly state any assumptions (e.g., 'Assumed rice-based dish due to visible grains'). If data is insufficient, produce a simple recipe and include a warning about potential inaccuracies.\n"
        f"2. A modified version of the same recipe{dietary_context}, ensuring all ingredients and methods meet the specified dietary needs (e.g., gluten-free noodles for gluten-free diets).\n"
        f"Do not add ingredients unless mentioned in captions, subtitles, audio, dish name, or video title. If unsure, prioritize simplicity and fidelity to provided data. Format both recipes consistently:\n\n"
        f"Visual descriptions:\n" + "\n".join(f"- {cap}" for cap in captions) + audio_context + title_context
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            full_response = response.choices[0].message.content
            
            recipes = full_response.split("\n\n###")
            if len(recipes) >= 2:
                original_recipe = recipes[0].strip()
                dietary_recipe = recipes[1].strip() if dietary_preference else ""
            else:
                original_recipe = ""
                dietary_recipe = ""
                sections = re.split(r'\n\s*(Ingredients|Instructions|Tips)\s*\n', full_response)
                current_recipe = []
                is_dietary = False
                for i, section in enumerate(sections):
                    section = section.strip()
                    if not section:
                        continue
                    if section in ["Ingredients", "Instructions", "Tips"]:
                        continue
                    current_recipe.append(section)
                    if len(current_recipe) >= 3:
                        if not is_dietary:
                            original_recipe = "\n\n".join(current_recipe)
                            current_recipe = []
                            is_dietary = True and dietary_preference
                        else:
                            dietary_recipe = "\n\n".join(current_recipe)
                            break
                if not original_recipe:
                    original_recipe = full_response.strip()
            
            if not original_recipe or "Ingredients" not in original_recipe or "Instructions" not in original_recipe:
                logger.warning(f"Attempt {attempt + 1}: Incomplete recipe generated.")
                if attempt == max_retries - 1:
                    raise Exception("Failed to generate complete recipe after retries.")
                continue
            
            logger.info(f"Original recipe generated: {original_recipe[:200]}...")
            if dietary_recipe:
                logger.info(f"Dietary recipe generated: {dietary_recipe[:200]}...")
            
            return original_recipe, dietary_recipe
        
        except Exception as e:
            logger.error(f"OpenAI API attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate recipe after {max_retries} attempts: {str(e)}")
            time.sleep(1)
    
    raise Exception("Failed to generate recipe: No valid response after retries.")

# === MAIN ROUTE ===
@app.route("/", methods=["GET", "POST"])
def index():
    logger.info(f"Handling request: {request.method} {request.url}")
    video_path = None
    audio_path = None
    subtitle_paths = []
    video_title = ""

    if not check_folder_permissions(UPLOAD_FOLDER):
        flash(f"Error: Cannot write to {UPLOAD_FOLDER}. Check folder permissions.", "error")
        return render_template_string(HTML_TEMPLATE)

    if request.method == "POST":
        try:
            # Option 1: File upload
            if "video" in request.files and request.files["video"].filename != "":
                file = request.files["video"]
                timestamp = int(time.time() * 1000)
                video_path = os.path.join(UPLOAD_FOLDER, f"uploaded_video_{timestamp}.mp4")
                file.save(video_path)
                if not os.path.exists(video_path):
                    raise Exception("Failed to save uploaded video.")
                video_title = file.filename  # Use filename as fallback title
                logger.info(f"Uploaded video saved to: {video_path}, Title: {video_title}")

            # Option 2: Video URL
            elif "video_url" in request.form and request.form["video_url"].strip():
                video_url = request.form["video_url"].strip()
                video_path, subtitle_paths, video_title = download_video_from_url(video_url)
                logger.info(f"Video downloaded from URL: {video_path}, Subtitles: {subtitle_paths}, Title: {video_title}")

            else:
                flash("Please upload a video or provide a valid video URL.", "error")
                return render_template_string(HTML_TEMPLATE)

            if not os.path.exists(video_path):
                raise Exception(f"Video file not found at {video_path}. Upload or download may have failed.")

            dish_name = request.form.get("dish_name", "").strip()
            dietary_preference = request.form.get("dietary_preference", "").strip()

            audio_transcription = ""
            audio_warning = ""
            if check_ffmpeg():
                timestamp = int(time.time() * 1000)
                audio_path = os.path.join(UPLOAD_FOLDER, f"audio_{timestamp}.mp3")
                try:
                    extract_audio(video_path, audio_path)
                    if os.path.exists(audio_path):
                        audio_transcription, audio_warning = transcribe_audio(audio_path, subtitle_paths)
                        if audio_warning:
                            flash(audio_warning, "warning")
                except Exception as e:
                    logger.warning(f"Audio processing failed: {str(e)}. Continuing without audio transcription.")
                    flash(f"Audio processing failed: {str(e)}. Relying on visual captions.", "warning")

            frames = extract_frames(video_path)
            captions = [caption_image(frame) for frame in frames]
            inferred_dish, inferred_cuisine = suggest_dish_name(captions, audio_transcription, video_title)
            filtered_captions = filter_captions(captions, inferred_cuisine)
            
            is_coherent, coherence_warning = check_caption_coherence(filtered_captions)
            if not is_coherent:
                flash(coherence_warning, "warning")
            
            if not dish_name and inferred_dish:
                dish_name = inferred_dish
                flash(f"Inferred dish: {dish_name}", "inferred_dish")
            
            if len(filtered_captions) < 3:
                flash("Limited high-quality captions may affect recipe accuracy.", "warning")
            
            if video_title:
                flash(video_title, "video_title")
            
            original_recipe, dietary_recipe = generate_recipe_from_captions(
                filtered_captions, audio_transcription, dish_name, dietary_preference, inferred_cuisine, video_title
            )

            flash(filtered_captions, "captions")
            flash(original_recipe, "recipe")
            if dietary_recipe:
                flash(dietary_recipe, "dietary_recipe")

            # Render the template directly with flashed messages instead of redirecting
            response = make_response(render_template_string(HTML_TEMPLATE, dish_name=dish_name))
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            logger.error(f"Error processing request: {str(e)}")

        finally:
            if not DEBUG_MODE:
                for file in os.listdir(UPLOAD_FOLDER):
                    file_path = os.path.join(UPLOAD_FOLDER, file)
                    if file_path not in [video_path, audio_path] + subtitle_paths and os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                            logger.info(f"Deleted old file: {file_path}")
                        except Exception as e:
                            logger.error(f"Failed to clean {file_path}: {str(e)}")
            if video_path and os.path.exists(video_path) and not DEBUG_MODE:
                try:
                    os.remove(video_path)
                    logger.info(f"Cleaned: {video_path}")
                except Exception as e:
                    logger.error(f"Failed to clean {video_path}: {str(e)}")
            if audio_path and os.path.exists(audio_path) and not DEBUG_MODE:
                try:
                    os.remove(audio_path)
                    logger.info(f"Cleaned: {audio_path}")
                except Exception as e:
                    logger.error(f"Failed to clean {audio_path}: {str(e)}")
            for subtitle_path in subtitle_paths:
                if subtitle_path and os.path.exists(subtitle_path) and not DEBUG_MODE:
                    try:
                        os.remove(subtitle_path)
                        logger.info(f"Cleaned: {subtitle_path}")
                    except Exception as e:
                        logger.error(f"Failed to clean {subtitle_path}: {str(e)}")

        # If an error occurs, render the template with flashed error messages
        response = make_response(render_template_string(HTML_TEMPLATE, dish_name=dish_name))
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    # For GET requests, render the template
    response = make_response(render_template_string(HTML_TEMPLATE))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# === RUN APP ===
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)