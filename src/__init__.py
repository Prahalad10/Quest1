"""DialogueFrame -- find the video frame where a line of dialogue is spoken.

Pipeline (src/service.py:find_dialogue): resolve -> audio -> probe -> index
-> match -> frame. Indexing is expensive and cached per video; querying is
cheap, which is why a repeat search skips ASR entirely.

The video is never downloaded: only the audio track is fetched, and the frame
is pulled with a ranged HTTP seek. The exception is a host that offers no
audio-only track, where the smallest progressive stream is used instead.
"""

__version__ = "0.1.0"
