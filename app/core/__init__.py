"""Core pipeline modules.

Each module here is independently runnable via `python -m app.core.<module>`.

Planned layout (built one step at a time):
    resolve.py   -- yt-dlp metadata + media_key + stream URLs
    audio.py     -- audio-only fetch, 16k mono wav, ffprobe of the video stream
    asr.py       -- faster-whisper transcription with word timestamps
    index.py     -- transcript index persistence + normalization
    matching.py  -- fuzzy substring match over the index
    frame.py     -- timestamp -> frame number -> ranged remote frame extract
"""
