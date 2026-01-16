# Video Subtitle Processor

A production-ready Python project that downloads streaming videos, transcribes French speech, translates to Turkish, and burns subtitles directly into the video.

## Features

- **Multi-source video support**: Downloads from m3u8, mp4, and YouTube URLs
- **Robust transcription**: Uses OpenAI Whisper with configurable models (tiny/base/small/medium/large)
- **Efficient translation**: Batch translation using deep-translator (reliable Google Translate API)
- **Flexible output**: Burn subtitles into video and/or save soft subtitle files (SRT)
- **Production-ready**: Comprehensive error handling, logging, and resource cleanup
- **Performance optimized**: Batch processing, reduced redundant operations, efficient file handling
- **Windows compatible**: Handles encoding issues and path problems automatically

## Prerequisites

1. **Python 3.8+** installed on your system
2. **FFmpeg** installed and available in your system PATH
   - Windows: `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)

## Installation

1. Clone or download this project
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

**Note**: The first time you run the script, Whisper will download the model (base model, ~150MB by default). This is a one-time download.

## Usage

### Command Line (Recommended)

```bash
# Basic usage
python video_subtitle_processor.py <video_url> [-o OUTPUT] [-m MODEL] [OPTIONS]

# Examples
python video_subtitle_processor.py https://www.youtube.com/watch?v=VIDEO_ID -o output.mp4

# Use smaller/faster model
python video_subtitle_processor.py https://example.com/video.m3u8 -m tiny -o output.mp4

# Generate soft subtitles only (no burning)
python video_subtitle_processor.py https://example.com/video.mp4 --no-burn --soft-subtitles

# Verbose logging
python video_subtitle_processor.py https://example.com/video.mp4 -v
```

### Command Line Options

```
positional arguments:
  video_url             URL of the video (m3u8, mp4, or YouTube)

optional arguments:
  -h, --help            Show help message
  -o, --output OUTPUT   Output filename (default: output_with_subtitles.mp4)
  -m, --model MODEL     Whisper model: tiny/base/small/medium/large (default: base)
  --no-burn             Don't burn subtitles, only generate SRT file
  --soft-subtitles      Also save soft subtitle file (SRT)
  --batch-size SIZE     Translation batch size (default: 50)
  -v, --verbose         Enable verbose logging
```

### Python API

```python
from video_subtitle_processor import VideoSubtitleProcessor

# Basic usage
with VideoSubtitleProcessor(whisper_model="base") as processor:
    results = processor.process_video(
        video_url="https://www.youtube.com/watch?v=VIDEO_ID",
        output_filename="output.mp4"
    )
    print(f"Original: {results['original']}")
    print(f"Final: {results['final']}")

# Advanced usage with options
with VideoSubtitleProcessor(
    whisper_model="small",
    burn_subtitles=True,
    soft_subtitles=True
) as processor:
    results = processor.process_video(
        video_url="https://example.com/video.m3u8",
        output_filename="my_video.mp4"
    )
```

## How It Works

1. **Video Download**: 
   - **YouTube**: Uses yt-dlp with best quality format selection
   - **Direct URLs (m3u8/mp4)**: Uses FFmpeg with proper HTTP headers for streaming
2. **Audio Extraction**: Extracts audio as WAV format (16kHz, mono) optimized for Whisper
3. **Transcription**: Uses OpenAI Whisper (configurable model) to transcribe French speech
4. **Translation**: Batch translates segments from French to Turkish using deep-translator (reliable Google Translate)
5. **SRT Generation**: Creates properly formatted SRT file with validated timestamps (HH:MM:SS,mmm)
6. **Subtitle Burning**: Uses FFmpeg to burn subtitles into video with styling (white text, black outline)
7. **Output**: Saves original video, final video with burned subtitles, and optionally soft subtitle file

## Output Structure

```
orijinalini/
  └── orijinal_output.mp4          # Original downloaded video

türkçe altyazı eklenmiş halini/
  └── output.mp4                   # Video with burned Turkish subtitles
  └── output.srt                   # Soft subtitle file (if --soft-subtitles used)
```

## Configuration

### Whisper Models

Choose based on your needs:
- **tiny**: Fastest, least accurate (~39MB model)
- **base**: Balanced (default, ~150MB model)
- **small**: Better accuracy (~500MB model)
- **medium**: High accuracy (~1.5GB model)
- **large**: Best accuracy (~3GB model)

### Translation Batch Size

Default is 50 segments per batch. Adjust with `--batch-size`:
- Smaller batches: More API calls, but more reliable
- Larger batches: Fewer API calls, but may hit rate limits

### Subtitle Styling

Edit the `subtitle_filter` in `burn_subtitles()` method to customize:
- Font size: `FontSize=24`
- Colors: `PrimaryColour=&Hffffff` (white), `OutlineColour=&H000000` (black)
- Outline: `Outline=2`

### Video Quality

Edit the `-crf` parameter in `burn_subtitles()`:
- Range: 18-28 (lower = better quality, larger file)
- Default: 23 (good balance)

## Performance Tips

1. **For long videos (1+ hour)**:
   - Use `tiny` or `small` Whisper model for faster transcription
   - Increase `--batch-size` for faster translation (if API allows)

2. **For better quality**:
   - Use `medium` or `large` Whisper model
   - Lower CRF value (e.g., 20) for better video quality

3. **For batch processing**:
   - Use the Python API in a loop
   - Reuse processor instance to avoid reloading Whisper model

## Troubleshooting

### FFmpeg not found
- Ensure FFmpeg is installed and accessible from command line
- Test by running: `ffmpeg -version`
- Windows: Restart terminal after installing FFmpeg

### YouTube download issues
- Ensure yt-dlp is installed: `pip install yt-dlp`
- Some YouTube videos may be age-restricted or region-locked
- Update yt-dlp: `pip install --upgrade yt-dlp`

### Translation errors
- The script uses deep-translator (Google Translate API) which may have rate limits
- If translation fails, the original French text will be used
- Reduce `--batch-size` if hitting rate limits

### Memory issues
- For very long videos, use a smaller Whisper model (`tiny` or `small`)
- Close other applications to free up memory

### Windows encoding issues
- The script automatically handles Windows path encoding issues
- If problems persist, ensure your terminal supports UTF-8

### Timeout errors
- Very long videos may timeout during processing
- Timeouts: Download (1h), Audio extraction (10m), Subtitle burning (2h)
- For extremely long videos, consider processing in segments

## Architecture & Design

### Key Improvements

1. **Translation Layer**: Replaced unstable googletrans with reliable deep-translator
2. **Batch Processing**: Efficient batch translation reduces API calls
3. **Timestamp Validation**: Proper SRT timestamp formatting with overflow handling
4. **Whisper Optimization**: Removed unnecessary word-level timestamps, configurable models
5. **FFmpeg Robustness**: Added HTTP headers, improved error handling, Windows path fixes
6. **Resource Management**: Context manager for automatic cleanup, proper error handling
7. **Logging**: Comprehensive logging for debugging and monitoring

### Code Structure

- **VideoSubtitleProcessor**: Main class with all processing logic
- **Context Manager**: Automatic resource cleanup
- **Error Handling**: Comprehensive exception handling with clear error messages
- **Logging**: Structured logging for production use

## License

This project is provided as-is for educational and personal use.
