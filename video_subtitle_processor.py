"""
Video Subtitle Processor
Downloads video, transcribes French speech, translates to Turkish,
generates SRT subtitles, and burns subtitles into the video.

Production-ready implementation with robust error handling,
batch processing, and performance optimizations.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from contextlib import contextmanager
import whisper
from deep_translator import GoogleTranslator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoSubtitleProcessor:
    """
    Main class for processing videos with subtitle generation and burning.
    
    Handles video download, audio extraction, transcription, translation,
    and subtitle burning with robust error handling and performance optimizations.
    """
    
    # Translation batch size for efficient API usage
    TRANSLATION_BATCH_SIZE = 50
    
    # Supported Whisper models (ordered by size/accuracy)
    WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']
    
    def __init__(
        self,
        output_dir: str = "output",
        whisper_model: str = "base",
        original_dir: str = "orijinalini",
        translated_dir: str = "türkçe altyazı eklenmiş halini",
        burn_subtitles: bool = True,
        soft_subtitles: bool = False
    ):
        """
        Initialize the processor.
        
        Args:
            output_dir: Directory to save output files
            whisper_model: Whisper model to use (tiny/base/small/medium/large)
            original_dir: Directory for original videos
            translated_dir: Directory for translated videos with subtitles
            burn_subtitles: Whether to burn subtitles into video (default: True)
            soft_subtitles: Whether to also save soft subtitle file (default: False)
        """
        if whisper_model not in self.WHISPER_MODELS:
            raise ValueError(f"Invalid Whisper model. Choose from: {self.WHISPER_MODELS}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.original_videos_dir = Path(original_dir)
        self.original_videos_dir.mkdir(exist_ok=True)
        self.translated_videos_dir = Path(translated_dir)
        self.translated_videos_dir.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_subtitle_"))
        self.whisper_model_name = whisper_model
        self.whisper_model = None
        self.translator = None
        self.burn_subtitles = burn_subtitles
        self.soft_subtitles = soft_subtitles
        
        logger.info(f"Initialized processor with Whisper model: {whisper_model}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temporary directory."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary directory safely."""
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
    
    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is installed and accessible."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def check_ytdlp(self) -> bool:
        """Check if yt-dlp is installed and accessible."""
        try:
            import yt_dlp
            return True
        except ImportError:
            return False
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL."""
        youtube_domains = ["youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com"]
        return any(domain in url.lower() for domain in youtube_domains)
    
    def _get_ffmpeg_headers(self) -> List[str]:
        """
        Get HTTP headers for FFmpeg to handle m3u8 streams.
        
        Returns:
            List of FFmpeg header arguments
        """
        return [
            "-headers", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "-headers", "Referer: https://www.example.com/"
        ]
    
    def download_video(self, video_url: str, output_filename: str) -> str:
        """
        Download video from URL using ffmpeg or yt-dlp (for YouTube).
        
        Args:
            video_url: URL of the video (m3u8, mp4, or YouTube)
            output_filename: Name for the downloaded video file
            
        Returns:
            Path to the downloaded video file
            
        Raises:
            RuntimeError: If download fails
        """
        logger.info(f"Downloading video from: {video_url}")
        
        # Check if it's a YouTube URL
        if self._is_youtube_url(video_url):
            return self._download_youtube_video(video_url, output_filename)
        
        # Use ffmpeg for direct video URLs (m3u8, mp4)
        output_path = self.temp_dir / output_filename
        
        # Build FFmpeg command with headers for m3u8 streams
        cmd = [
            "ffmpeg",
            *self._get_ffmpeg_headers(),
            "-i", video_url,
            "-c", "copy",  # Copy streams without re-encoding (faster)
            "-y",  # Overwrite output file
            "-loglevel", "warning",  # Reduce log verbosity
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=3600  # 1 hour timeout
            )
            logger.info(f"Video downloaded successfully: {output_path}")
            return str(output_path)
        except subprocess.TimeoutExpired:
            error_msg = "Video download timed out after 1 hour"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8', errors='ignore')
            logger.error(f"Error downloading video: {error_msg}")
            raise RuntimeError(f"Failed to download video: {error_msg}")
    
    def _download_youtube_video(self, video_url: str, output_filename: str) -> str:
        """
        Download video from YouTube using yt-dlp.
        
        Args:
            video_url: YouTube video URL
            output_filename: Name for the downloaded video file
            
        Returns:
            Path to the downloaded video file
            
        Raises:
            ImportError: If yt-dlp is not installed
            RuntimeError: If download fails
        """
        try:
            import yt_dlp
        except ImportError:
            raise ImportError(
                "yt-dlp is required for YouTube downloads. "
                "Install it with: pip install yt-dlp"
            )
        
        logger.info("Detected YouTube URL, using yt-dlp...")
        output_path = self.temp_dir / output_filename
        
        # Configure yt-dlp options for best quality
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(output_path.with_suffix('.%(ext)s')),
            'merge_output_format': 'mp4',
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Find the downloaded file
            downloaded_file = output_path.with_suffix('.mp4')
            if not downloaded_file.exists():
                video_files = list(self.temp_dir.glob('*.mp4'))
                if video_files:
                    downloaded_file = max(video_files, key=lambda p: p.stat().st_mtime)
                else:
                    raise FileNotFoundError("Downloaded video file not found")
            
            # Rename to desired output filename if needed
            if downloaded_file != output_path:
                downloaded_file.rename(output_path)
            
            logger.info(f"YouTube video downloaded successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error downloading YouTube video: {e}")
            raise RuntimeError(f"Failed to download YouTube video: {e}")
    
    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file optimized for Whisper.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the extracted audio file
            
        Raises:
            RuntimeError: If extraction fails
        """
        logger.info("Extracting audio from video...")
        audio_path = self.temp_dir / "audio.wav"
        
        # Extract audio as WAV format optimized for Whisper
        # 16kHz mono is optimal for Whisper
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit little-endian
            "-ar", "16000",  # Sample rate 16kHz (optimal for Whisper)
            "-ac", "1",  # Mono channel
            "-y",
            "-loglevel", "warning",
            str(audio_path)
        ]
        
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600  # 10 minute timeout
            )
            logger.info(f"Audio extracted successfully: {audio_path}")
            return str(audio_path)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio extraction timed out")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8', errors='ignore')
            logger.error(f"Error extracting audio: {error_msg}")
            raise RuntimeError(f"Failed to extract audio: {error_msg}")
    
    def _load_whisper_model(self):
        """Load Whisper model if not already loaded (lazy loading)."""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {self.whisper_model_name} (this may take a moment)...")
            try:
                self.whisper_model = whisper.load_model(self.whisper_model_name)
                logger.info(f"Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def transcribe_audio(self, audio_path: str, language: str = "fr") -> dict:
        """
        Transcribe audio using OpenAI Whisper.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (default: "fr" for French)
            
        Returns:
            Dictionary containing transcription results with segments and timestamps
            
        Raises:
            RuntimeError: If transcription fails
        """
        logger.info("Transcribing audio with Whisper...")
        
        # Load model if needed
        self._load_whisper_model()
        
        try:
            # Transcribe WITHOUT word-level timestamps (not needed, saves memory)
            result = self.whisper_model.transcribe(
                audio_path,
                language=language,
                word_timestamps=False,  # Not needed for SRT generation
                verbose=False
            )
            
            logger.info("Transcription completed.")
            return result
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}")
    
    def _init_translator(self):
        """Initialize translator if not already initialized."""
        if self.translator is None:
            try:
                self.translator = GoogleTranslator(source='fr', target='tr')
                logger.debug("Translator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize translator: {e}")
                raise RuntimeError(f"Failed to initialize translator: {e}")
    
    def _translate_batch(self, texts: List[str], max_retries: int = 3) -> List[str]:
        """
        Translate a batch of texts efficiently using true network-level batching.
        
        Joins multiple subtitle segments into a single string, makes ONE translation
        API call per batch, then splits the result back into individual segments.
        This dramatically reduces API calls and avoids rate limiting.
        
        Args:
            texts: List of texts to translate
            max_retries: Maximum number of retry attempts per batch
            
        Returns:
            List of translated texts in the same order as input
        """
        if not texts:
            return []
        
        self._init_translator()
        
        # Track which positions have empty texts (to preserve them)
        empty_positions = {i for i, text in enumerate(texts) if not text or not text.strip()}
        
        # Filter out empty texts and track their original positions
        non_empty_texts = []
        position_map = []  # Maps index in non_empty_texts to original index
        
        for i, text in enumerate(texts):
            if i not in empty_positions:
                non_empty_texts.append(text.strip())
                position_map.append(i)
        
        # If all texts are empty, return as-is
        if not non_empty_texts:
            return texts
        
        # Join all texts with newline delimiter
        # Using newline as delimiter is safe for subtitles (unlikely to appear in text)
        joined_text = "\n".join(non_empty_texts)
        
        # Translate the entire batch in ONE API call
        translated_joined = None
        for attempt in range(max_retries):
            try:
                translated_joined = self.translator.translate(joined_text)
                if translated_joined:
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"Batch translation attempt {attempt + 1} failed, retrying...")
                    continue
                else:
                    logger.warning(f"Batch translation failed after {max_retries} attempts: {e}")
                    # Fall back to per-segment translation
                    return self._translate_batch_fallback(texts, max_retries)
        
        # If translation failed, fall back to per-segment
        if not translated_joined:
            logger.warning("Batch translation returned empty result, falling back to per-segment translation")
            return self._translate_batch_fallback(texts, max_retries)
        
        # Split the translated result back by newlines
        translated_lines = translated_joined.split("\n")
        
        # Validate line count matches
        if len(translated_lines) != len(non_empty_texts):
            logger.warning(
                f"Translated line count ({len(translated_lines)}) doesn't match "
                f"input count ({len(non_empty_texts)}), falling back to per-segment translation"
            )
            return self._translate_batch_fallback(texts, max_retries)
        
        # Map translated lines back to original positions
        result = [""] * len(texts)
        for translated_idx, original_idx in enumerate(position_map):
            result[original_idx] = translated_lines[translated_idx].strip()
        
        # Preserve empty strings in their original positions
        for empty_idx in empty_positions:
            result[empty_idx] = texts[empty_idx]
        
        return result
    
    def _translate_batch_fallback(self, texts: List[str], max_retries: int = 3) -> List[str]:
        """
        Fallback method: translate texts one by one.
        
        Used when batch translation fails or returns mismatched line counts.
        
        Args:
            texts: List of texts to translate
            max_retries: Maximum number of retry attempts per text
            
        Returns:
            List of translated texts
        """
        logger.debug("Using fallback per-segment translation")
        self._init_translator()
        
        translated = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                translated.append(text)
                continue
            
            text = text.strip()
            translated_text = None
            
            for attempt in range(max_retries):
                try:
                    translated_text = self.translator.translate(text)
                    if translated_text:
                        break
                except Exception as e:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.warning(f"Translation failed for segment {i+1}: {e}")
            
            translated.append(translated_text if translated_text else text)
        
        return translated
    
    def generate_srt(
        self,
        transcription: dict,
        output_path: str,
        batch_size: int = None
    ) -> str:
        """
        Generate SRT subtitle file from transcription with Turkish translations.
        
        Uses batch translation for efficiency.
        
        Args:
            transcription: Whisper transcription result dictionary
            output_path: Path to save the SRT file
            batch_size: Number of segments to translate in each batch (default: TRANSLATION_BATCH_SIZE)
            
        Returns:
            Path to the generated SRT file
        """
        logger.info("Generating SRT file with Turkish translations...")
        
        if batch_size is None:
            batch_size = self.TRANSLATION_BATCH_SIZE
        
        srt_path = Path(output_path)
        segments = transcription.get("segments", [])
        total_segments = len(segments)
        
        if not segments:
            logger.warning("No segments found in transcription")
            # Create empty SRT file
            srt_path.touch()
            return str(srt_path)
        
        # Prepare all texts for batch translation
        segment_texts = [segment.get("text", "").strip() for segment in segments]
        
        # Translate in batches
        logger.info(f"Translating {total_segments} segments in batches of {batch_size}...")
        translated_texts = []
        
        for i in range(0, len(segment_texts), batch_size):
            batch = segment_texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(segment_texts) + batch_size - 1) // batch_size
            
            logger.info(f"Translating batch {batch_num}/{total_batches} ({len(batch)} segments)...")
            translated_batch = self._translate_batch(batch)
            translated_texts.extend(translated_batch)
        
        # Write SRT file
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, (segment, translated_text) in enumerate(zip(segments, translated_texts), start=1):
                start_time = self._format_timestamp(segment["start"])
                end_time = self._format_timestamp(segment["end"])
                
                # Write SRT entry
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{translated_text}\n")
                f.write("\n")
        
        logger.info(f"SRT file generated: {srt_path}")
        return str(srt_path)
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds to SRT timestamp format (HH:MM:SS,mmm).
        
        Ensures proper formatting and handles edge cases:
        - Milliseconds overflow (>= 1000)
        - Negative values
        - Very large values
        
        Args:
            seconds: Time in seconds (float)
            
        Returns:
            Formatted timestamp string in HH:MM:SS,mmm format
        """
        # Ensure non-negative
        seconds = max(0.0, float(seconds))
        
        # Calculate components
        total_seconds = int(seconds)
        milliseconds = int((seconds - total_seconds) * 1000)
        
        # Handle millisecond overflow
        if milliseconds >= 1000:
            total_seconds += milliseconds // 1000
            milliseconds = milliseconds % 1000
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        # Format: HH:MM:SS,mmm
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _prepare_srt_path_for_ffmpeg(self, srt_path: str) -> str:
        """
        Prepare SRT file path for FFmpeg on Windows.
        
        Handles encoding issues by copying to ASCII-only path if needed.
        
        Args:
            srt_path: Original SRT file path
            
        Returns:
            Escaped path suitable for FFmpeg
        """
        srt_path_obj = Path(srt_path)
        
        if os.name == 'nt':  # Windows
            # Copy to temp dir with simple ASCII name to avoid encoding issues
            simple_srt_path = self.temp_dir / "subs.srt"
            shutil.copy2(srt_path, simple_srt_path)
            srt_abs_path = simple_srt_path.absolute()
            # Use forward slashes and escape colons for FFmpeg
            srt_escaped = str(srt_abs_path).replace("\\", "/").replace(":", "\\:")
        else:
            srt_abs_path = srt_path_obj.absolute()
            srt_escaped = str(srt_abs_path)
        
        return srt_escaped
    
    def _burn_subtitles_into_video(self, video_path: str, srt_path: str, output_path: str) -> str:
        """
        Burn subtitles into video using FFmpeg.
        
        Args:
            video_path: Path to the input video file
            srt_path: Path to the SRT subtitle file
            output_path: Path to save the output video
            
        Returns:
            Path to the output video with burned subtitles
            
        Raises:
            RuntimeError: If subtitle burning fails
        """
        logger.info("Burning subtitles into video...")
        
        output_path = Path(output_path)
        
        # Handle Windows encoding issues
        if os.name == 'nt' and any(ord(c) > 127 for c in str(output_path)):
            temp_output = self.temp_dir / f"output_{output_path.name}"
            final_output_path = output_path
            output_path = temp_output
            logger.debug(f"Using temporary path for encoding compatibility: {temp_output}")
        else:
            final_output_path = None
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare SRT path for FFmpeg
        srt_escaped = self._prepare_srt_path_for_ffmpeg(srt_path)
        
        # Build FFmpeg command for subtitle burning
        # Windows requires proper escaping: use single quotes around path, escape single quotes in path
        if os.name == 'nt':  # Windows
            # Escape single quotes in path and wrap in single quotes
            srt_escaped_quoted = srt_escaped.replace("'", "'\\''")
            subtitle_filter = (
                f"subtitles='{srt_escaped_quoted}':"
                f"force_style='FontSize=24,PrimaryColour=&Hffffff,"
                f"OutlineColour=&H000000,Outline=2'"
            )
        else:
            subtitle_filter = (
                f"subtitles={srt_escaped}:"
                f"force_style='FontSize=24,PrimaryColour=&Hffffff,"
                f"OutlineColour=&H000000,Outline=2'"
            )
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", subtitle_filter,
            "-c:v", "libx264",
            "-c:a", "copy",  # Copy audio without re-encoding
            "-preset", "medium",  # Balance between speed and compression
            "-crf", "23",  # Good quality (18-28 range, lower = better)
            "-y",
            "-loglevel", "warning",
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=7200  # 2 hour timeout for long videos
            )
            
            # Copy to final location if we used temp path
            if final_output_path:
                final_output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_path, final_output_path)
                logger.info(f"Subtitles burned successfully: {final_output_path}")
                return str(final_output_path)
            else:
                logger.info(f"Subtitles burned successfully: {output_path}")
                return str(output_path)
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Subtitle burning timed out after 2 hours")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if isinstance(e.stderr, str) else e.stderr.decode('utf-8', errors='ignore')
            logger.error(f"Error burning subtitles: {error_msg}")
            raise RuntimeError(f"Failed to burn subtitles: {error_msg}")
    
    def process_video(
        self,
        video_url: str,
        output_filename: str = "output_with_subtitles.mp4"
    ) -> Dict[str, str]:
        """
        Complete pipeline: download, transcribe, translate, and burn subtitles.
        
        Args:
            video_url: URL of the video to process
            output_filename: Name for the final output file
            
        Returns:
            Dictionary with paths to:
            - 'original': Original video path
            - 'subtitle': SRT subtitle file path (if soft_subtitles=True)
            - 'final': Final video with burned subtitles (if burn_subtitles=True)
            
        Raises:
            RuntimeError: If any step fails
        """
        logger.info("=" * 60)
        logger.info("Starting video subtitle processing pipeline")
        logger.info("=" * 60)
        
        # Validate dependencies
        if not self.check_ffmpeg():
            raise RuntimeError("ffmpeg is not installed or not in PATH. Please install ffmpeg.")
        
        if self._is_youtube_url(video_url) and not self.check_ytdlp():
            raise RuntimeError(
                "yt-dlp is required for YouTube downloads. "
                "Install it with: pip install yt-dlp"
            )
        
        results = {}
        
        try:
            # Step 1: Download video
            video_path = self.download_video(video_url, "downloaded_video.mp4")
            
            # Save original video
            original_filename = f"orijinal_{Path(output_filename).stem}.mp4"
            original_save_path = self.original_videos_dir / original_filename
            shutil.copy2(video_path, original_save_path)
            results['original'] = str(original_save_path)
            logger.info(f"Original video saved to: {original_save_path}")
            
            # Step 2: Extract audio
            audio_path = self.extract_audio(video_path)
            
            # Step 3: Transcribe audio (French)
            transcription = self.transcribe_audio(audio_path, language="fr")
            
            # Step 4: Generate SRT file with Turkish translations
            srt_path = self.temp_dir / "subtitles.srt"
            self.generate_srt(transcription, srt_path)
            
            # Save soft subtitle file if requested
            if self.soft_subtitles:
                soft_subtitle_path = self.translated_videos_dir / f"{Path(output_filename).stem}.srt"
                shutil.copy2(srt_path, soft_subtitle_path)
                results['subtitle'] = str(soft_subtitle_path)
                logger.info(f"Soft subtitle file saved to: {soft_subtitle_path}")
            
            # Step 5: Burn subtitles into video (if requested)
            if self.burn_subtitles:
                final_output = self.translated_videos_dir / output_filename
                final_output.parent.mkdir(parents=True, exist_ok=True)
                final_video_path = self._burn_subtitles_into_video(video_path, str(srt_path), str(final_output))
                results['final'] = final_video_path
                logger.info(f"Translated video with subtitles saved to: {final_video_path}")
            
            logger.info("=" * 60)
            logger.info("Processing complete!")
            for key, path in results.items():
                logger.info(f"  {key.capitalize()}: {path}")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
            raise


def main():
    """Main entry point for the script with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Download video, transcribe French speech, translate to Turkish, and burn subtitles"
    )
    parser.add_argument(
        "video_url",
        help="URL of the video (m3u8, mp4, or YouTube)"
    )
    parser.add_argument(
        "-o", "--output",
        default="output_with_subtitles.mp4",
        help="Output filename (default: output_with_subtitles.mp4)"
    )
    parser.add_argument(
        "-m", "--model",
        choices=VideoSubtitleProcessor.WHISPER_MODELS,
        default="base",
        help="Whisper model to use (default: base)"
    )
    parser.add_argument(
        "--no-burn",
        action="store_true",
        help="Don't burn subtitles, only generate SRT file"
    )
    parser.add_argument(
        "--soft-subtitles",
        action="store_true",
        help="Also save soft subtitle file (SRT)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=VideoSubtitleProcessor.TRANSLATION_BATCH_SIZE,
        help=f"Translation batch size (default: {VideoSubtitleProcessor.TRANSLATION_BATCH_SIZE})"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create processor with context manager for automatic cleanup
    with VideoSubtitleProcessor(
        whisper_model=args.model,
        burn_subtitles=not args.no_burn,
        soft_subtitles=args.soft_subtitles
    ) as processor:
        try:
            results = processor.process_video(args.video_url, args.output)
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            for key, path in results.items():
                print(f"{key.capitalize()}: {path}")
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
