"""Funzioni di utilità per elaborazione e confronto video"""
import tempfile
import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def get_video_info(video_path):
    """Estrae informazioni base dal video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration,
    }


def extract_frames_from_videos(video_paths_dict, original_video_path, num_frames=100):
    """
    Estrae frame sincronizzati da tutti i video per confronto frame-by-frame.

    Args:
        video_paths_dict: dict con {nome_metodo: path_video}
        original_video_path: path del video originale
        num_frames: numero di frame da estrarre uniformemente

    Returns:
        dict con {nome_video: list_of_frames}
    """
    all_videos = {'📹 Originale': original_video_path}
    all_videos.update(video_paths_dict)

    frames_dict = {}

    for name, video_path in all_videos.items():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        frames_dict[name] = frames

    return frames_dict


def combine_videos_grid(video_paths, titles, output_path=None, max_video_width=640):
    """
    Combina multiple video in una griglia sincronizzata con titoli.

    Args:
        video_paths: lista dei path dei video da combinare
        titles: lista dei titoli per ogni video
        output_path: path di output (opzionale)
        max_video_width: larghezza massima per ogni video nella griglia

    Returns:
        path del video combinato o None se fallisce
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix='_comparison_grid.mp4')

    caps = [cv2.VideoCapture(str(vp)) for vp in video_paths]
    if not all(c.isOpened() for c in caps):
        for cap in caps:
            cap.release()
        return None

    fps = caps[0].get(cv2.CAP_PROP_FPS)
    original_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    if original_width > max_video_width:
        scale_factor = max_video_width / original_width
        width = max_video_width
        height = int(original_height * scale_factor)
    else:
        width = original_width
        height = original_height

    n = len(video_paths)
    cols = 2 if n == 2 else 3
    rows = (n + cols - 1) // cols

    title_height = 35
    padding = 8
    cell_width = width
    cell_height = height + title_height

    grid_width = cell_width * cols + padding * (cols + 1)
    grid_height = cell_height * rows + padding * (rows + 1)

    max_grid_dimension = 2560
    if grid_width > max_grid_dimension or grid_height > max_grid_dimension:
        scale = min(max_grid_dimension / grid_width, max_grid_dimension / grid_height)
        width = int(width * scale)
        height = int(height * scale)
        cell_width = width
        cell_height = height + title_height
        grid_width = cell_width * cols + padding * (cols + 1)
        grid_height = cell_height * rows + padding * (rows + 1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace('.mp4', '.avi')
        out = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
        if not out.isOpened():
            for cap in caps:
                cap.release()
            return None

    frame_count = 0
    max_frames = min([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps])

    while frame_count < max_frames:
        frames = []
        all_ok = True
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                all_ok = False
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(frame)

        if not all_ok:
            break

        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 40

        for idx, (frame, title) in enumerate(zip(frames, titles)):
            row = idx // cols
            col = idx % cols

            x_start = padding + col * (cell_width + padding)
            y_start = padding + row * (cell_height + padding)

            title_area_y_end = y_start + title_height
            grid[y_start:title_area_y_end, x_start:x_start + cell_width] = [30, 30, 30]

            # Titolo con PIL
            title_region = cv2.cvtColor(
                grid[y_start:title_area_y_end, x_start:x_start + cell_width],
                cv2.COLOR_BGR2RGB
            )
            title_pil = Image.fromarray(title_region)
            draw = ImageDraw.Draw(title_pil)

            font = None
            for font_path in [
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
            ]:
                try:
                    font = ImageFont.truetype(font_path, size=20)
                    break
                except Exception:
                    continue
            if font is None:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), title, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = (cell_width - text_w) // 2
            text_y = (title_height - text_h) // 2

            draw.text((text_x + 1, text_y + 1), title, font=font, fill=(0, 0, 0))
            draw.text((text_x, text_y), title, font=font, fill=(255, 255, 255))

            grid[y_start:title_area_y_end, x_start:x_start + cell_width] = cv2.cvtColor(
                np.array(title_pil), cv2.COLOR_RGB2BGR
            )

            frame_y_start = y_start + title_height
            grid[frame_y_start:frame_y_start + height, x_start:x_start + width] = frame

        out.write(grid)
        frame_count += 1

    for cap in caps:
        cap.release()
    out.release()

    if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
        return None
    return output_path


def convert_to_web_compatible(input_path, output_path):
    """Converte il video in H.264 compatibile con i browser usando FFmpeg bundled."""
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    try:
        result = subprocess.run(
            [
                ffmpeg_exe,
                "-y",                  # sovrascrive senza chiedere
                "-i", input_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",
                "-an",                 # nessun audio
                output_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0 and Path(output_path).exists()
    except Exception as e:
        st.warning(f"Conversione FFmpeg non riuscita: {e}")
        return False
