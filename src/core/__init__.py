"""Pipeline stages, each runnable on its own: python -m src.core.<module> --help

Running one stage at a time is how you tell a mishearing from a match failure.

    resolve    yt-dlp metadata and stream selection (the only yt-dlp caller)
    audio      audio -> 16kHz mono wav, plus ffprobe of the video stream
    ffmpeg     subprocess wrapper for ffmpeg/ffprobe (not a stage)
    asr        faster-whisper transcription with word timestamps
    normalize  the shared text folding used at index AND query time
    index      transcript cache: flat text + char->word offsets
    matching   fuzzy search over that index
    frame      timestamp -> frame number -> PNG via ranged seek

Nothing here knows about HTTP or the CLI; orchestration is src/service.py.
"""
