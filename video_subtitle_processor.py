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
import glob
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from contextlib import contextmanager

# Try to import OpenAI
try:
    from openai import OpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

# Try to import DeepL translator
try:
    from deep_translator import DeeplTranslator  # type: ignore
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False
    DeeplTranslator = None  # type: ignore

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip .env loading

# Try to import faster-whisper (faster, GPU-accelerated) first, fallback to whisper
try:
    from faster_whisper import WhisperModel  # type: ignore
    FASTER_WHISPER_AVAILABLE = True
    WHISPER_AVAILABLE = False
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    try:
        import whisper  # type: ignore
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False

# Try to import torch for GPU detection
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPT translation system prompt (shared constant to avoid duplication)
GPT_SYSTEM_PROMPT = (
    "You are translating subtitles for a French TV series.\n\n"
    "Translate from French to Turkish.\n"
    "Preserve meaning and context, not word-for-word translation.\n"
    "Adapt expressions naturally to Turkish.\n"
    "Do NOT explain.\n"
    "Do NOT add or remove lines.\n"
    "Keep the exact line order.\n"
    "Keep sentences short and readable for subtitles.\n"
    "Use a serious, cinematic tone (Netflix / spy-drama style)."
)


class VideoSubtitleProcessor:
    """
    Main class for processing videos with subtitle generation and burning.
    
    Handles video download, audio extraction, transcription, translation,
    and subtitle burning with robust error handling and performance optimizations.
    """
    
    # Translation batch size for efficient API usage (character-based for GPT)
    TRANSLATION_BATCH_SIZE = 100  # Legacy: kept for backward compatibility
    MAX_CHARS_PER_BATCH = 12000  # Character limit per batch (safe for GPT)
    
    # Supported Whisper models (ordered by size/accuracy)
    WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']
    
    def __init__(
        self,
        output_dir: str = "output",
        whisper_model: str = "base",
        original_dir: str = "orijinalini",
        translated_dir: str = "türkçe altyazı eklenmiş halini",
        burn_subtitles: bool = True,
        soft_subtitles: bool = False,
        openai_api_key: str = None,
        deepl_api_key: str = None,
        translator: str = "gpt"
    ):
        # Translator backend selection
        if translator not in ["deepl", "gpt"]:
            raise ValueError(f"Invalid translator. Choose from: 'deepl' or 'gpt'")
        self.translator_backend = translator
        
        # OpenAI API key (required for GPT translation)
        if openai_api_key is None:
            openai_api_key = os.environ.get('OPENAI_API_KEY')
        
        # DeepL API key (required for DeepL translation)
        if deepl_api_key is None:
            deepl_api_key = os.environ.get('DEEPL_API_KEY')
        
        """
        Initialize the processor.
        
        Args:
            output_dir: Directory to save output files
            whisper_model: Whisper model to use (tiny/base/small/medium/large)
            original_dir: Directory for original videos
            translated_dir: Directory for translated videos with subtitles
            burn_subtitles: Whether to burn subtitles into video (default: True)
            soft_subtitles: Whether to also save soft subtitle file (default: False)
            openai_api_key: OpenAI API key for translation (required when translator='gpt')
            deepl_api_key: DeepL API key for translation (required when translator='deepl')
            translator: Translation backend ('deepl' or 'gpt', default: 'deepl')
        """
        if whisper_model not in self.WHISPER_MODELS:
            raise ValueError(f"Invalid Whisper model. Choose from: {self.WHISPER_MODELS}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.original_videos_dir = Path(original_dir)
        self.original_videos_dir.mkdir(exist_ok=True)
        self.translated_videos_dir = Path(translated_dir)
        self.translated_videos_dir.mkdir(exist_ok=True)
        
        # GPT translated videos directory
        self.gpt_translated_dir = Path("gpt-translate")
        self.gpt_translated_dir.mkdir(exist_ok=True)
        
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_subtitle_"))
        self.whisper_model_name = whisper_model
        self.whisper_model = None
        self.translator = None
        self.burn_subtitles = burn_subtitles
        self.soft_subtitles = soft_subtitles
        self.openai_api_key = openai_api_key
        self.deepl_api_key = deepl_api_key
        
        # Silence gap threshold for subtitle freeze fix
        self.SILENCE_GAP_THRESHOLD = 1.5  # seconds
        
        # Track if GPT cost has been logged (to avoid spam)
        self._gpt_cost_logged = False
        
        logger.info(f"Using translator: {self.translator_backend.upper()}")
        
        # Detect GPU availability for optimization
        self.use_gpu = self._detect_gpu()
        # Use faster-whisper if available (works on both CPU and GPU, but GPU is faster)
        self.use_faster_whisper = FASTER_WHISPER_AVAILABLE
        
        if self.use_faster_whisper:
            logger.info(f"Initialized with Faster Whisper (GPU accelerated) - model: {whisper_model}")
        elif FASTER_WHISPER_AVAILABLE:
            logger.info(f"Initialized with Faster Whisper (CPU) - model: {whisper_model}")
        else:
            logger.info(f"Initialized with standard Whisper - model: {whisper_model}")
    
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
            # Try to find FFmpeg in common Windows locations (WinGet installation)
            if os.name == 'nt':  # Windows
                common_paths = [
                    os.path.join(os.environ.get('LOCALAPPDATA', ''), 
                                'Microsoft', 'WinGet', 'Packages', 
                                'Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe',
                                'ffmpeg-*', 'bin', 'ffmpeg.exe'),
                    r'C:\ffmpeg\bin\ffmpeg.exe',
                    os.path.join(os.environ.get('ProgramFiles', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
                ]
                
                for pattern in common_paths:
                    matches = glob.glob(pattern)
                    if matches:
                        ffmpeg_path = matches[0]
                        try:
                            # Add to PATH for this process
                            bin_dir = os.path.dirname(ffmpeg_path)
                            if bin_dir not in os.environ.get('PATH', ''):
                                os.environ['PATH'] = f"{bin_dir};{os.environ.get('PATH', '')}"
                            # Test if it works
                            subprocess.run(
                                ["ffmpeg", "-version"],
                                capture_output=True,
                                check=True,
                                timeout=5
                            )
                            logger.info(f"FFmpeg found at: {ffmpeg_path}")
                            return True
                        except:
                            continue
            
            return False
    
    def check_ytdlp(self) -> bool:
        """Check if yt-dlp is installed and accessible."""
        try:
            import yt_dlp  # type: ignore
            return True
        except ImportError:
            return False
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available for acceleration."""
        if not TORCH_AVAILABLE:
            return False
        try:
            return torch.cuda.is_available()
        except:
            return False
    
    def _detect_hw_encoder(self) -> Optional[str]:
        """
        Detect available hardware encoder for FFmpeg.
        
        Returns:
            Hardware encoder name (h264_nvenc, h264_qsv, h264_videotoolbox) or None
        """
        if not self.check_ffmpeg():
            return None
        
        try:
            # Check for NVIDIA NVENC
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=5
            )
            encoders = result.stdout.lower()
            
            if "h264_nvenc" in encoders and self.use_gpu:
                return "h264_nvenc"
            elif "h264_qsv" in encoders:
                return "h264_qsv"  # Intel Quick Sync
            elif "h264_videotoolbox" in encoders:
                return "h264_videotoolbox"  # macOS
        except:
            pass
        
        return None
    
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
            import yt_dlp  # type: ignore
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
        
        # Use cookies.txt if provided
        if hasattr(self, "cookies_path") and self.cookies_path:
            ydl_opts["cookies"] = self.cookies_path
            logger.info(f"Using cookies file: {self.cookies_path}")
        
        # Try to use cookies from browser for age-restricted videos
        # This allows downloading age-restricted content without manual cookie export
        browsers_to_try = ['chrome', 'edge', 'firefox', 'opera', 'brave']
        cookies_used = False
        cookie_error_message = None
        
        for browser in browsers_to_try:
            try:
                # Create test options with cookies
                test_opts = ydl_opts.copy()
                test_opts['cookiesfrombrowser'] = (browser,)
                test_opts['quiet'] = True  # Suppress output during test
                
                # Try to initialize - this will fail if cookies can't be loaded
                try:
                    test_ydl = yt_dlp.YoutubeDL(test_opts)
                    # Test cookie loading by trying to get cookiejar (this will trigger cookie load)
                    _ = test_ydl.cookiejar
                    del test_ydl
                    # If we got here, cookies are loaded successfully
                    ydl_opts['cookiesfrombrowser'] = (browser,)
                    logger.info(f"Using cookies from {browser} for age-restricted videos")
                    cookies_used = True
                    break
                except Exception as cookie_err:
                    # Cookie loading failed for this browser, try next
                    error_msg = str(cookie_err).lower()
                    if 'cookie' in error_msg or 'permission' in error_msg:
                        cookie_error_message = f"Cookie access failed for {browser}: {cookie_err}"
                        logger.debug(cookie_error_message)
                    continue
            except Exception as e:
                # Browser not available, try next one
                continue
        
        if not cookies_used:
            if cookie_error_message and 'permission' in cookie_error_message.lower():
                logger.warning("⚠️  Cookie access failed - Chrome/Edge may be open. Close browser and retry for age-restricted videos.")
            else:
                logger.info("ℹ️  No browser cookies available - age-restricted videos may fail")
        
        # Try download with cookies first, fallback to without cookies if it fails
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
            error_str = str(e)
            # Check if it's a cookie-related error
            if 'cookie' in error_str.lower() or 'CookieLoadError' in error_str or 'permission denied' in error_str.lower():
                logger.warning(f"Cookie loading failed: {error_str}")
                
                # Check if this is an age-restricted video error
                if 'sign in to confirm your age' in error_str.lower() or 'age-restricted' in error_str.lower():
                    logger.error("=" * 60)
                    logger.error("❌ AGE-RESTRICTED VIDEO - COOKIES REQUIRED")
                    logger.error("=" * 60)
                    logger.error("This video requires authentication. To fix:")
                    logger.error("1. Close Chrome/Edge/Firefox browser completely")
                    logger.error("2. Run the script again")
                    logger.error("3. Or manually export cookies:")
                    logger.error("   - Install 'Get cookies.txt' Chrome extension")
                    logger.error("   - Export cookies.txt file")
                    logger.error("   - Use: python script.py --cookies cookies.txt")
                    logger.error("=" * 60)
                    raise RuntimeError(
                        "Age-restricted video requires cookies. "
                        "Close your browser and try again, or export cookies manually."
                    )
                
                logger.info("Retrying without cookies...")
                
                # Remove cookies and try again
                if 'cookiesfrombrowser' in ydl_opts:
                    del ydl_opts['cookiesfrombrowser']
                
                # Retry download without cookies
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
                    
                    logger.info(f"YouTube video downloaded successfully (without cookies): {output_path}")
                    return str(output_path)
                except Exception as retry_e:
                    retry_error = str(retry_e)
                    # Check if still age-restricted error
                    if 'sign in to confirm your age' in retry_error.lower() or 'age-restricted' in retry_error.lower():
                        logger.error("=" * 60)
                        logger.error("❌ AGE-RESTRICTED VIDEO - COOKIES REQUIRED")
                        logger.error("=" * 60)
                        logger.error("This video requires authentication. To fix:")
                        logger.error("1. Close Chrome/Edge/Firefox browser completely")
                        logger.error("2. Run the script again")
                        logger.error("3. Or manually export cookies from browser")
                        logger.error("=" * 60)
                        raise RuntimeError(
                            "Age-restricted video requires cookies. "
                            "Close your browser and try again, or export cookies manually."
                        )
                    logger.error(f"Error downloading YouTube video (retry without cookies): {retry_e}")
                    raise RuntimeError(f"Failed to download YouTube video: {retry_e}")
            else:
                # Non-cookie error, raise normally
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
        # Use multi-threading for faster extraction
        cmd = [
                "ffmpeg",
                "-threads", "0",  # Use all CPU threads
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
        if self.whisper_model is not None:
            return
        
        logger.info(f"Loading Whisper model: {self.whisper_model_name} (this may take a moment)...")
        
        if self.use_faster_whisper:
            try:
                # Use faster-whisper with GPU if available
                device = "cuda" if self.use_gpu else "cpu"
                compute_type = "float16" if self.use_gpu else "int8"  # float16 on GPU, int8 on CPU for speed
                
                self.whisper_model = WhisperModel(
                    self.whisper_model_name,
                    device=device,
                    compute_type=compute_type,
                    num_workers=4 if not self.use_gpu else 1  # Multi-threading on CPU
                )
                logger.info(f"Faster Whisper model loaded successfully ({device}, {compute_type})")
                return
            except Exception as e:
                logger.warning(f"Failed to load faster-whisper, falling back to standard whisper: {e}")
                self.use_faster_whisper = False
        
        # Fallback to standard whisper
        if not WHISPER_AVAILABLE:
            raise RuntimeError(
                "Neither faster-whisper nor whisper is installed. "
                "Install one with: pip install faster-whisper OR pip install openai-whisper"
            )
        
        try:
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            logger.info("Standard Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def transcribe_audio(self, audio_path: str, language: str = "fr") -> dict:
        """
        Transcribe audio using Whisper (faster-whisper if available, else standard).
        
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
            if self.use_faster_whisper:
                # Use faster-whisper API (much faster, especially on GPU)
                segments, info = self.whisper_model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=5,  # Balanced speed/accuracy
                    vad_filter=True,  # Voice activity detection for better accuracy
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Convert to standard Whisper format
                segments_list = []
                for segment in segments:
                    segments_list.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip()
                    })
                
                result = {
                    "text": " ".join([s["text"] for s in segments_list]),
                    "language": info.language,
                    "segments": segments_list
                }
            else:
                # Use standard Whisper API
                result = self.whisper_model.transcribe(
                    audio_path,
                    language=language,
                    word_timestamps=False,  # Not needed for SRT generation
                    verbose=False,
                    fp16=self.use_gpu  # Use FP16 on GPU for speed
                )
            
            logger.info("Transcription completed.")
            return result
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}")
    
    def _init_translator(self):
        """Initialize translator (DeepL or GPT) if not already initialized."""
        if self.translator is None:
            if self.translator_backend == "deepl":
                try:
                    if not DEEPL_AVAILABLE:
                        raise ImportError(
                            "DeepL package is not installed. "
                            "Install it with: pip install deep-translator"
                        )
                    if not self.deepl_api_key:
                        raise ValueError(
                            "DeepL API key is required. "
                            "Get one from https://www.deepl.com/pro-api or set DEEPL_API_KEY environment variable."
                        )
                    # Free API key formatı ':fx' ile biter
                    is_free_api = self.deepl_api_key.endswith(':fx')
                    
                    self.translator = DeeplTranslator(
                        api_key=self.deepl_api_key,
                        source='fr',
                        target='tr',
                        use_free_api=is_free_api  # Otomatik algıla: free API key ise True
                    )
                    logger.debug("DeepL translator initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize DeepL translator: {e}")
                    raise RuntimeError(f"Failed to initialize DeepL translator: {e}")
            elif self.translator_backend == "gpt":
                try:
                    if not OPENAI_AVAILABLE:
                        raise ImportError(
                            "OpenAI package is not installed. "
                            "Install it with: pip install openai"
                        )
                    if not self.openai_api_key:
                        raise ValueError(
                            "OpenAI API key is required. "
                            "Get one from https://platform.openai.com/api-keys or set OPENAI_API_KEY environment variable."
                        )
                    
                    self.translator = OpenAI(api_key=self.openai_api_key)
                    logger.debug("OpenAI translator initialized with model: gpt-5-mini")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI translator: {e}")
                    raise RuntimeError(f"Failed to initialize OpenAI translator: {e}")
    
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
        
        # Join all texts with numbered format to prevent line mismatch
        # Format: [1] text1\n[2] text2\n[3] text3
        # This ensures GPT preserves exact line count even if it modifies newlines
        numbered_texts = [f"[{i+1}] {text}" for i, text in enumerate(non_empty_texts)]
        joined_text = "\n".join(numbered_texts)
        
        # Translate the entire batch in ONE API call
        translated_joined = None
        for attempt in range(max_retries):
            try:
                if self.translator_backend == "deepl":
                    # DeepL translation
                    translated_joined = self.translator.translate(joined_text)
                elif self.translator_backend == "gpt":
                    # GPT translation with Responses API
                    # Log cost estimation only once to avoid spam
                    if not self._gpt_cost_logged:
                        logger.info("Using GPT-5-mini (~$0.05 per hour of video)")
                        self._gpt_cost_logged = True
                    
                    response = self.translator.responses.create(
                        model="gpt-5-mini",
                        # temperature parameter removed - GPT-5-mini doesn't support it
                        input=[
                            {"role": "system", "content": GPT_SYSTEM_PROMPT},
                            {"role": "user", "content": f"TEXT:\n{joined_text}\n\nPreserve the [number] format exactly."}
                        ]
                    )
                    
                    # Log actual token usage if available
                    if hasattr(response, "usage") and response.usage:
                        logger.info(
                            f"GPT usage | input={response.usage.input_tokens} "
                            f"output={response.usage.output_tokens} "
                            f"total={response.usage.input_tokens + response.usage.output_tokens}"
                        )
                    
                    translated_joined = response.output_text.strip()
                
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
        
        # Extract numbered lines using regex: [1] text, [2] text, etc.
        # This is more reliable than newline splitting because GPT might modify newlines
        # Pattern matches: [1] text, [2] text, etc. (handles multi-line text between brackets)
        line_pattern = r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|$)'
        matches = re.findall(line_pattern, translated_joined, re.DOTALL | re.MULTILINE)
        
        # Validate we got the expected number of lines
        if len(matches) != len(non_empty_texts):
            logger.error(
                f"GPT line mismatch: input={len(non_empty_texts)}, output={len(matches)}. "
                "Falling back to per-segment translation."
            )
            return self._translate_batch_fallback(texts, max_retries)
        
        # Sort by line number and extract text (in case GPT reorders)
        sorted_matches = sorted(matches, key=lambda x: int(x[0]))
        translated_texts_by_number = {int(num): text.strip() for num, text in sorted_matches}
        
        # Map translated lines back to original positions
        result = [""] * len(texts)
        for translated_idx, original_idx in enumerate(position_map):
            line_num = translated_idx + 1
            if line_num in translated_texts_by_number:
                result[original_idx] = translated_texts_by_number[line_num]
            else:
                logger.warning(f"Missing line {line_num} in GPT response, using original")
                result[original_idx] = non_empty_texts[translated_idx]
        
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
                    if self.translator_backend == "deepl":
                        # DeepL translation
                        translated_text = self.translator.translate(text)
                    elif self.translator_backend == "gpt":
                        # GPT translation with Responses API
                        response = self.translator.responses.create(
                            model="gpt-5-mini",
                            # temperature parameter removed - GPT-5-mini doesn't support it
                            input=[
                                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                                {"role": "user", "content": f"TEXT:\n{text}"}
                            ]
                        )
                        
                        # Log actual token usage if available
                        if hasattr(response, "usage") and response.usage:
                            logger.debug(
                                f"GPT usage (fallback) | input={response.usage.input_tokens} "
                                f"output={response.usage.output_tokens} "
                                f"total={response.usage.input_tokens + response.usage.output_tokens}"
                            )
                        
                        translated_text = response.output_text.strip()
                    
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
        
        # Create character-based batches (more reliable for GPT token limits)
        batches = []
        current_batch = []
        current_chars = 0
        
        for text in segment_texts:
            text_chars = len(text)
            
            # If adding this text would exceed limit, start new batch
            # Account for newline characters between segments
            newline_chars = len(current_batch)  # One newline per existing segment
            if current_batch and (current_chars + text_chars + newline_chars) > self.MAX_CHARS_PER_BATCH:
                batches.append(current_batch)
                current_batch = [text]
                current_chars = text_chars
            else:
                current_batch.append(text)
                current_chars += text_chars
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        # Translate in character-based batches
        total_batches = len(batches)
        logger.info(f"Translating {total_segments} segments in {total_batches} character-based batches (max {self.MAX_CHARS_PER_BATCH} chars/batch)...")
        translated_texts = []
        
        for batch_num, batch in enumerate(batches, 1):
            batch_chars = sum(len(text) for text in batch) + len(batch) - 1  # + newlines
            logger.info(f"Translating batch {batch_num}/{total_batches} ({len(batch)} segments, ~{batch_chars} chars)...")
            translated_batch = self._translate_batch(batch)
            translated_texts.extend(translated_batch)
        
        # Two-pass approach for silence freeze fix and overlap prevention
        # First pass: Calculate adjusted end times
        adjusted_ends = []
        for i, segment in enumerate(segments):
            original_end = segment["end"]
            adjusted_end = original_end
            
            # Check for long silence gap after this segment
            if i < len(segments) - 1:
                next_start = segments[i + 1]["start"]
                gap = next_start - original_end
                
                if gap > self.SILENCE_GAP_THRESHOLD:
                    # Close subtitle early to prevent freeze during silence
                    adjusted_end = next_start - 0.1
                    logger.debug(f"Closing subtitle {i+1} early due to {gap:.2f}s silence gap")
            
            # Prevent overlap with next segment
            if i < len(segments) - 1:
                next_start = segments[i + 1]["start"]
                if adjusted_end > next_start:
                    adjusted_end = next_start - 0.05
                    logger.debug(f"Adjusted subtitle {i+1} end time to prevent overlap")
            
            adjusted_ends.append(adjusted_end)
        
        # Second pass: Write SRT file with adjusted times
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, (segment, translated_text, adjusted_end) in enumerate(zip(segments, translated_texts, adjusted_ends), start=1):
                start_time = self._format_timestamp(segment["start"])
                end_time = self._format_timestamp(adjusted_end)
                
                # Post-process translation for better subtitle tone
                translated_text = self._improve_translation(translated_text)
                
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
    
    def _improve_translation(self, text: str) -> str:
        """
        Post-process translation based on translator backend.
        
        For GPT: Light cleanup only (whitespace normalization, punctuation fixes)
        For DeepL: Aggressive processing for espionage/drama content style
        
        Args:
            text: Translated text
            
        Returns:
            Improved text
        """
        if not text or not text.strip():
            return text
        
        result = text.strip()
        
        # Light cleanup for both translators
        result = re.sub(r'\s+', ' ', result).strip()  # Normalize whitespace
        
        # GPT: Only light cleanup, no aggressive processing
        if self.translator_backend == "gpt":
            # Basic punctuation fixes
            result = re.sub(r'\s+([,.!?;:])', r'\1', result)  # Remove space before punctuation
            return result
        
        # DeepL: Aggressive processing (original behavior)
        # 1. Remove polite phrases
        polite_phrases = ["Lütfen", "Endişelenmeyin", "Sorun değil", "Merhaba", "İyi günler"]
        for phrase in polite_phrases:
            result = re.sub(rf'\b{phrase}\b[,\.\s]*', '', result, flags=re.IGNORECASE)
        
        # 2. Remove leading pronouns (Ben, Biz, O, Onlar)
        result = re.sub(r'^(Ben|Biz|O|Onlar)\s+', '', result, flags=re.IGNORECASE)
        
        # 3. Remove exclamation marks
        result = result.replace('!', '')
        result = result.replace('¡', '')
        
        # 4. Convert short questions (≤6 words) to statements
        words = result.split()
        if len(words) <= 6 and result.strip().endswith('?'):
            result = result.rstrip('?') + '.'
        
        # 5. Remove time fillers
        time_fillers = ["şu an", "şu anda", "hemen", "şimdi", "bir an"]
        for filler in time_fillers:
            result = re.sub(rf'\b{filler}\b[\s,]*', '', result, flags=re.IGNORECASE)
        
        # 6. Apply spy terminology dictionary
        spy_dict = {
            'legend': 'kimlik hikâyesi',
            'asset': 'eleman',
            'cover': 'paravan',
            'handler': 'saha sorumlusu',
            'extraction': 'tahliye'
        }
        for en, tr in spy_dict.items():
            result = re.sub(rf'\b{en}\b', tr, result, flags=re.IGNORECASE)
        
        # Recalculate words after all removals/replacements
        words = result.split()
        
        # 7. Add tension to very short sentences (≤3 words)
        if len(words) <= 3 and not result.endswith('…'):
            result = result.rstrip('.!?') + '…'
        
        # 8. Shorten very long sentences (12+ words) - keep first 10 words
        if len(words) > 12:
            result = ' '.join(words[:10]) + '…'
        
        # Clean up multiple spaces and trailing punctuation
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
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
        
        # Build FFmpeg command for subtitle burning with Netflix-like styling
        # Windows requires proper escaping: use single quotes around path, escape single quotes in path
        # Netflix-like style: larger font, shadow, margins for better readability
        style_params = (
            "FontSize=28,"  # Larger font for TV/laptop viewing
            "PrimaryColour=&Hffffff,"  # White text
            "OutlineColour=&H000000,"  # Black outline
            "Outline=2,"  # Outline thickness
            "Shadow=1,"  # Enable shadow
            "ShadowX=2,"  # Shadow X offset
            "ShadowY=2,"  # Shadow Y offset
            "MarginV=30"  # Bottom margin for better positioning
        )
        
        if os.name == 'nt':  # Windows
            # Escape single quotes in path and wrap in single quotes
            srt_escaped_quoted = srt_escaped.replace("'", "'\\''")
            subtitle_filter = (
                f"subtitles='{srt_escaped_quoted}':"
                f"force_style='{style_params}'"
            )
        else:
            subtitle_filter = (
                f"subtitles={srt_escaped}:"
                f"force_style='{style_params}'"
            )
        
        # Detect hardware encoder for faster encoding
        hw_encoder = self._detect_hw_encoder()
        
        # Try hardware encoder first, fallback to software if it fails
        if hw_encoder:
            logger.info(f"Trying hardware encoder: {hw_encoder} for faster encoding")
            # Build command with hardware encoder
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", subtitle_filter,
                "-c:v", hw_encoder,
                "-c:a", "copy",  # Copy audio without re-encoding
            ]
            
            # Add encoder-specific options BEFORE output file
            if hw_encoder == "h264_nvenc":
                cmd.extend(["-preset", "p4", "-cq", "23"])  # NVIDIA: p4=fast, p1=fastest
            elif hw_encoder == "h264_qsv":
                cmd.extend(["-global_quality", "23"])  # Intel QSV
            elif hw_encoder == "h264_videotoolbox":
                cmd.extend(["-quality", "70"])  # macOS VideoToolbox
            
            # Add output file and other options
            cmd.extend(["-y", "-loglevel", "warning", str(output_path)])
            
            # Try hardware encoder first
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
                
                # Success with hardware encoder
                if final_output_path:
                    final_output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(output_path, final_output_path)
                    logger.info(f"Subtitles burned successfully (hardware): {final_output_path}")
                    return str(final_output_path)
                else:
                    logger.info(f"Subtitles burned successfully (hardware): {output_path}")
                    return str(output_path)
                    
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                # Hardware encoder failed, fallback to software
                logger.warning(f"Hardware encoder {hw_encoder} failed, falling back to software encoder: {e}")
                hw_encoder = None  # Force software encoder
        
        # Use software encoder (either because no HW available or HW failed)
        if not hw_encoder:
            logger.info("Using software encoder (libx264)")
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", subtitle_filter,
                "-c:v", "libx264",
                "-c:a", "copy",  # Copy audio without re-encoding
                "-preset", "ultrafast",  # Fastest preset
                "-crf", "23",  # Good quality (18-28 range, lower = better)
                "-threads", "0",  # Use all available CPU threads
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
                # Select output directory based on translator
                if self.translator_backend == "gpt":
                    soft_dir = self.gpt_translated_dir
                else:
                    soft_dir = self.translated_videos_dir
                
                soft_subtitle_path = soft_dir / f"{Path(output_filename).stem}.srt"
                shutil.copy2(srt_path, soft_subtitle_path)
                results['subtitle'] = str(soft_subtitle_path)
                logger.info(f"Soft subtitle file saved to: {soft_subtitle_path}")
            
            # Step 5: Burn subtitles into video (if requested)
            if self.burn_subtitles:
                # Select output directory based on translator
                if self.translator_backend == "gpt":
                    output_dir = self.gpt_translated_dir
                else:
                    output_dir = self.translated_videos_dir
                
                final_output = output_dir / output_filename
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
        nargs="?",
        help="URL of the video (m3u8, mp4, or YouTube). If not provided, will be asked interactively."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output filename. If not provided, will be asked interactively."
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
        help=f"Translation batch size in segments (legacy, character-based batching is used for GPT, default: {VideoSubtitleProcessor.TRANSLATION_BATCH_SIZE})"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--translator",
        choices=["deepl", "gpt"],
        default="gpt",  # GPT-5-mini as default, DeepL must be explicitly selected
        help="Translation backend: 'deepl' or 'gpt' (default: gpt)"
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="OpenAI API key for translation. If not provided, will use OPENAI_API_KEY environment variable."
    )
    parser.add_argument(
        "--deepl-api-key",
        default=None,
        help="DeepL API key for translation. If not provided, will use DEEPL_API_KEY environment variable."
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Path to cookies.txt file for YouTube age-restricted videos"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get API keys based on translator selection
    openai_api_key = args.openai_api_key or os.environ.get('OPENAI_API_KEY')
    deepl_api_key = args.deepl_api_key or os.environ.get('DEEPL_API_KEY')
    
    # Validate API key based on translator selection
    if args.translator == "gpt":
        if not openai_api_key:
            print("❌ Hata: OpenAI API key gerekli!", file=sys.stderr)
            print("   Lütfen aşağıdaki yöntemlerden birini kullanın:", file=sys.stderr)
            print("   1. --openai-api-key parametresi ile: --openai-api-key YOUR_KEY", file=sys.stderr)
            print("   2. OPENAI_API_KEY environment variable ile", file=sys.stderr)
            print("   API key almak için: https://platform.openai.com/api-keys", file=sys.stderr)
            sys.exit(1)
    elif args.translator == "deepl":
        if not deepl_api_key:
            print("❌ Hata: DeepL API key gerekli!", file=sys.stderr)
            print("   Lütfen aşağıdaki yöntemlerden birini kullanın:", file=sys.stderr)
            print("   1. --deepl-api-key parametresi ile: --deepl-api-key YOUR_KEY", file=sys.stderr)
            print("   2. DEEPL_API_KEY environment variable ile", file=sys.stderr)
            print("   API key almak için: https://www.deepl.com/pro-api", file=sys.stderr)
            sys.exit(1)
    
    # Interactively collect video list if not provided
    video_url = args.video_url
    output_filename = args.output
    
    if not video_url:
        print("=" * 60)
        print("Video Subtitle Processor - İnteraktif Mod")
        print("=" * 60)
        print("\n📝 Birden fazla video işlemek için sırayla URL ve isim girin.")
        print("   Tüm videoları girdikten sonra URL kısmını boş bırakıp Enter'a basın.\n")
        
        videos = []
        video_num = 1
        
        while True:
            url = input(f"📹 Video {video_num} URL'ini girin (boş bırakıp Enter = bitir): ").strip()
            if not url:
                break
            
            name = input(f"💾 Video {video_num} için çıktı dosya adı: ").strip()
            if not name:
                print("⚠️  İsim boş olamaz, tekrar deneyin.")
                continue
            
            # Add .mp4 extension if not present
            if not name.lower().endswith('.mp4'):
                name += '.mp4'
            
            videos.append({'url': url, 'name': name})
            video_num += 1
            print()
        
        if not videos:
            print("❌ Hata: Hiç video girilmedi!")
            sys.exit(1)
        
        # Show summary
        print("\n" + "=" * 60)
        print(f"✅ Toplam {len(videos)} video eklendi:")
        for i, video in enumerate(videos, 1):
            print(f"   {i}. {video['name']}")
        print("=" * 60)
        
        # Confirm before processing
        input("\n🚀 İşleme başlamak için Enter'a basın...")
        print()
        
        # Process all videos
        successful = 0
        failed = 0
        
        cookies_path = args.cookies
        
        with VideoSubtitleProcessor(
            whisper_model=args.model,
            burn_subtitles=not args.no_burn,
            soft_subtitles=args.soft_subtitles,
            openai_api_key=openai_api_key,
            deepl_api_key=deepl_api_key,
            translator=args.translator
        ) as processor:
            processor.cookies_path = cookies_path
            for i, video in enumerate(videos, 1):
                print("\n" + "=" * 60)
                print(f"📹 Video {i}/{len(videos)} işleniyor: {video['name']}")
                print("=" * 60)
                print(f"   URL: {video['url']}")
                print()
                
                try:
                    results = processor.process_video(video['url'], video['name'])
                    print(f"\n✅ Video {i} başarıyla işlendi: {video['name']}")
                    successful += 1
                except KeyboardInterrupt:
                    print(f"\n⚠️  İşlem kullanıcı tarafından durduruldu.")
                    print(f"   Video {i} işlenemedi: {video['name']}")
                    failed += 1
                    raise  # Re-raise to allow user to stop all processing
                except Exception as e:
                    logger.error(f"Video {i} işlenirken hata: {e}", exc_info=True)
                    print(f"\n❌ Video {i} işlenirken hata oluştu: {video['name']}")
                    print(f"   Hata: {str(e)}")
                    print(f"   ⏭️  Diğer videolara geçiliyor...\n")
                    failed += 1
                    continue  # Continue with next video
        
        # Final summary
        print("\n" + "=" * 60)
        print("📊 İŞLEM ÖZETİ")
        print("=" * 60)
        print(f"   ✅ Başarılı: {successful}")
        print(f"   ❌ Başarısız: {failed}")
        print(f"   📹 Toplam: {len(videos)}")
        print("=" * 60)
        
        # Don't exit on failures - just show summary
        # All videos were attempted, continue normally
        if successful == 0:
            print("\n⚠️  Hiçbir video işlenemedi!")
            sys.exit(1)
        elif failed > 0:
            print(f"\n⚠️  {failed} video işlenemedi, ancak {successful} video başarıyla tamamlandı.")
        
        return
    
    # Single video mode (original behavior)
    if not output_filename:
        output_filename = input("💾 Çıktı dosya adını girin (örn: video_tr.mp4) [Enter = output_with_subtitles.mp4]: ").strip()
        if not output_filename:
            output_filename = "output_with_subtitles.mp4"
        # Add .mp4 extension if not present
        if not output_filename.lower().endswith('.mp4'):
            output_filename += '.mp4'
    
    print(f"\n🚀 İşlem başlatılıyor...")
    print(f"   URL: {video_url}")
    print(f"   Çıktı: {output_filename}")
    translator_name = "GPT (gpt-5-mini)" if args.translator == "gpt" else "DeepL"
    print(f"   Çeviri: {translator_name}\n")
    
    # Create processor with context manager for automatic cleanup
    cookies_path = args.cookies
    
    with VideoSubtitleProcessor(
        whisper_model=args.model,
        burn_subtitles=not args.no_burn,
        soft_subtitles=args.soft_subtitles,
        openai_api_key=openai_api_key,
        deepl_api_key=deepl_api_key,
        translator=args.translator
    ) as processor:
        processor.cookies_path = cookies_path
        try:
            results = processor.process_video(video_url, output_filename)
            print("\n" + "=" * 60)
            print("✅ BAŞARILI!")
            print("=" * 60)
            for key, path in results.items():
                print(f"   {key.capitalize()}: {path}")
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
