"""
Simple Flask web interface for Video Subtitle Processor.

This is a thin wrapper around the existing VideoSubtitleProcessor pipeline.
The pipeline code is NOT modified - it's imported and used as-is.
"""

import os
import sys
import uuid
import threading
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template

# Import the existing pipeline (black-box, no modifications)
# Add parent directory to path to import the pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))
from video_subtitle_processor import VideoSubtitleProcessor

app = Flask(__name__)

# Simple in-memory job tracking
# In production, use a proper database or Redis
jobs = {}

# Progress tracking: job_id → percent (0-100)
progress = {}

# Track start times for time estimation
start_times = {}


def process_video_background(job_id: str, video_url: str):
    """
    Background task to process video using the existing pipeline.
    
    This function calls the existing VideoSubtitleProcessor.process_video()
    method without any modifications to the pipeline code.
    """
    start_time = time.time()
    start_times[job_id] = start_time  # Track start time for time estimation
    
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['message'] = 'Starting video processing...'
        progress[job_id] = 10  # Starting
        
        # Time-based progress estimation thread (since pipeline is black-box)
        def update_progress_estimate():
            while jobs[job_id]['status'] == 'processing':
                # Check if already completed (race condition protection)
                if progress.get(job_id, 0) >= 100:
                    break
                elapsed = time.time() - start_time
                if elapsed < 30:
                    progress[job_id] = 10 + int((elapsed / 30) * 20)  # 10-30% in first 30s
                elif elapsed < 120:
                    progress[job_id] = 30 + int(((elapsed - 30) / 90) * 30)  # 30-60% in next 90s
                elif elapsed < 300:
                    progress[job_id] = 60 + int(((elapsed - 120) / 180) * 20)  # 60-80% in next 3min
                else:
                    # After 5 minutes, slowly increase to 95% max (leave room for completion)
                    extra_time = elapsed - 300
                    progress[job_id] = min(95, 80 + int(extra_time / 12))  # 80-95% over next 3 minutes
                time.sleep(2)  # Update every 2 seconds
        
        # Start progress estimation thread
        progress_thread = threading.Thread(target=update_progress_estimate, daemon=True)
        progress_thread.start()
        
        # Call the existing pipeline as-is (black-box)
        with VideoSubtitleProcessor() as processor:
            output_filename = f"output_{job_id}.mp4"
            results = processor.process_video(video_url, output_filename)
            
            # Get the final video path from results
            final_video_path = results.get('final')
            
            if final_video_path and Path(final_video_path).exists():
                # Set status first to stop progress thread, then set progress to 100%
                jobs[job_id]['status'] = 'completed'
                progress[job_id] = 100  # Complete
                jobs[job_id]['file'] = final_video_path
                jobs[job_id]['message'] = 'Processing completed successfully!'
            else:
                jobs[job_id]['status'] = 'error'
                jobs[job_id]['error'] = 'Video file not found after processing'
                jobs[job_id]['message'] = 'Processing failed: Video file not found'
                
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['message'] = f'Processing failed: {str(e)}'


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """
    Start video processing.
    
    Input: JSON with 'video_url' field
    Returns: JSON with 'job_id' for status checking
    """
    data = request.get_json()
    
    if not data or 'video_url' not in data:
        return jsonify({'error': 'video_url is required'}), 400
    
    video_url = data['video_url'].strip()
    
    if not video_url:
        return jsonify({'error': 'video_url cannot be empty'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    jobs[job_id] = {
        'status': 'queued',
        'message': 'Job queued, starting processing...',
        'file': None,
        'error': None
    }
    
    # Initialize progress
    progress[job_id] = 0
    
    # Start background processing thread
    thread = threading.Thread(
        target=process_video_background,
        args=(job_id, video_url),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'message': 'Processing started'
    })


@app.route('/status/<job_id>')
def status(job_id):
    """
    Check processing status.
    
    Returns: JSON with status information
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'message': job.get('message', ''),
        'error': job.get('error')
    })


@app.route('/progress/<job_id>')
def get_progress(job_id):
    """
    Get progress percentage for a job.
    
    Returns: JSON with percent (0-100) and time_remaining (in seconds)
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    percent = progress.get(job_id, 0)
    
    # Calculate estimated time remaining
    time_remaining = 0
    if percent > 0 and percent < 100:
        # Estimate total time based on current progress
        # Simple linear estimation
        elapsed = time.time() - start_times.get(job_id, time.time())
        if percent > 0:
            estimated_total = elapsed / (percent / 100)
            time_remaining = max(0, estimated_total - elapsed)
    
    return jsonify({
        'percent': percent,
        'time_remaining': int(time_remaining)
    })


@app.route('/download/<job_id>')
def download(job_id):
    """
    Download the processed video file.
    
    Returns: Video file if processing is completed
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({
            'error': 'Processing not completed',
            'status': job['status']
        }), 400
    
    file_path = job.get('file')
    
    if not file_path or not Path(file_path).exists():
        return jsonify({'error': 'Video file not found'}), 404
    
    # Send file for download
    return send_file(
        file_path,
        as_attachment=True,
        download_name=f"translated_video_{job_id}.mp4",
        mimetype='video/mp4'
    )


if __name__ == '__main__':
    # Run Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)

