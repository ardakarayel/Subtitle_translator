"""
Basit kullanım örneği - Video URL'ini gir ve çalıştır
"""

from video_subtitle_processor import VideoSubtitleProcessor

# Video URL'inizi buraya girin
VIDEO_URL = "YOUR_VIDEO_URL_HERE"  # Örnek: "https://www.youtube.com/watch?v=VIDEO_ID"

# İşlemi başlat
print("🚀 Video işleniyor...")
print(f"📹 URL: {VIDEO_URL}")
print()

with VideoSubtitleProcessor(
    whisper_model="base",  # tiny, base, small, medium, large
    burn_subtitles=True,   # Altyazıyı videoya göm
    soft_subtitles=False   # SRT dosyası kaydetme
) as processor:
    results = processor.process_video(VIDEO_URL, "output_video.mp4")
    
    print("\n" + "="*60)
    print("✅ İşlem tamamlandı!")
    print("="*60)
    print(f"📁 Final video: {results.get('final', 'N/A')}")
    print(f"📁 Orijinal video: {results.get('original', 'N/A')}")

