"""Pipeline stages. Each module here is independently runnable and testable.

    python -m app.core.<module> --help

WHY EACH STAGE IS SEPARATELY RUNNABLE
    Debugging a wrong answer means asking which stage produced it: did ASR
    mishear the line, or did matching fail to find text that was transcribed
    correctly? Running one stage at a time answers that in seconds instead of
    re-running the whole pipeline.

THE MODULES, IN PIPELINE ORDER
    resolve.py    yt-dlp metadata, stream URL selection, media_key, refusals.
                  The ONLY module that talks to yt-dlp.
    audio.py      audio-only fetch -> 16kHz mono wav; ffprobe of the video
                  stream -> probe.json (fps, VFR flag, dimensions).
    ffmpeg.py     shared subprocess wrapper for ffmpeg/ffprobe. Not a stage.
    asr.py        faster-whisper transcription with word-level timestamps.
                  The expensive stage.
    normalize.py  THE shared text normalization function. Used at index time
                  and at query time; the two must never diverge.
    index.py      builds and caches the transcript index: flat normalized text
                  plus the char->word offset array.
    matching.py   fuzzy search over that index, confidence bands, near misses.
    frame.py      timestamp -> frame number, and a ranged HTTP seek -> PNG.

SEPARATION OF CONCERNS
    resolve/audio/asr/index are INDEXING -- expensive, once per video.
    matching/frame are QUERYING -- cheap, once per search.
    That split is what lets a second search on the same video skip ASR.

    Nothing in this package knows about HTTP, FastAPI, or the CLI. Orchestration
    lives in app/service.py.
"""
