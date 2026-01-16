# Web Interface for Video Subtitle Processor

Simple Flask web interface for the video subtitle processing pipeline.

## Setup

1. Install Flask (if not already installed):
```bash
pip install -r requirements.txt
```

2. Make sure all pipeline dependencies are installed (from root `requirements.txt`):
```bash
cd ..
pip install -r requirements.txt
```

## Running the Web App

From the `web_app` directory:

```bash
python app.py
```

Or from the project root:

```bash
cd web_app
python app.py
```

The web interface will be available at: `http://localhost:5000`

## Usage

1. Open `http://localhost:5000` in your browser
2. Paste a video URL (YouTube, m3u8, or mp4)
3. Click "Process Video"
4. Wait for processing to complete (this may take several minutes)
5. Download the translated video when ready

## Architecture

- **Backend**: Flask app (`app.py`) that wraps the existing `VideoSubtitleProcessor`
- **Frontend**: Simple HTML page with JavaScript for status polling
- **Processing**: Background threads handle video processing
- **Storage**: In-memory job tracking (simple dict)

## Important Notes

- The existing pipeline code (`video_subtitle_processor.py`) is **NOT modified**
- The pipeline is imported and used as a black-box
- Processing happens in background threads (non-blocking)
- Job status is tracked in memory (lost on server restart)

## API Endpoints

- `GET /` - Main page
- `POST /process` - Start video processing (requires `video_url` in JSON)
- `GET /status/<job_id>` - Check processing status
- `GET /download/<job_id>` - Download completed video

