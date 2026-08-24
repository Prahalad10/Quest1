"""Quest1 -- locate the exact video frame where a line of dialogue is spoken.

Given a video URL and a line of text, the pipeline finds WHEN that line is
spoken and extracts the video frame at that moment -- without ever downloading
the video.

LAYOUT
    app/config.py    all tunable constants, every one env-overridable
    app/errors.py    the Quest1Error hierarchy every caller catches
    app/paths.py     the on-disk cache layout, in one place
    app/progress.py  the progress-callback contract shared by all stages
    app/service.py   find_dialogue() -- the one entry point to the pipeline
    app/cli.py       terminal front end (a thin wrapper over service.py)
    app/core/        the pipeline stages, each independently runnable

WHERE TO START READING
    app/service.py shows the whole pipeline in one function. Each stage it calls
    lives in app/core/ and can be run on its own from the command line.
"""

__version__ = "0.1.0"
