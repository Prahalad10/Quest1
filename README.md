# DialogueFrame — find the frame where a line of dialogue is spoken

Give it a video URL and a line of dialogue. It returns the exact timestamp, the
frame number, and a PNG of that frame.

```
Timestamp : 00:05:25.120
Frame     : 7795
Text      : "My mind rebels at stagnation."
Image     : outputs/the-adventures-of-sherlock-holmes-...-c02e1b060a/frames/frame_000325120.png
```

The search runs **on the audio only** — the dialogue is located by transcribing
speech, never by reading text off the screen. The video itself is not
downloaded: the frame is pulled with a ranged HTTP seek against the remote
stream. Supported hosts: **YouTube** and **ok.ru**.

---

## 1. Setup

Requires **Python 3.11+** and **ffmpeg** (a system dependency — pip cannot
install it).

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

Install ffmpeg if you do not have it:

| OS | Command |
|---|---|
| Windows | `winget install --id=Gyan.FFmpeg -e --source winget` |
| macOS | `brew install ffmpeg` |
| Linux | `sudo apt update && sudo apt install -y ffmpeg` |

Then confirm the environment is complete:

```bash
python scripts/check_env.py
```

It prints a pass/fail table for the Python version, `ffmpeg`, `ffprobe`, and
each third-party import, and exits non-zero if anything is missing.

---

## 2. Run it

### Command line

```bash
python -m src.cli "<video_url>" "<dialogue text>"
```

Examples:

```bash
python -m src.cli "https://www.youtube.com/watch?v=jNQXAC9IVRw" "they have really really long trunks"
python -m src.cli "https://ok.ru/video/248244667877" "My mind rebels at stagnation"
```

| Flag | Effect |
|---|---|
| `--json` | Machine-readable output on stdout |
| `--quiet` | Suppress the progress stream on stderr |
| `--force` | Ignore every cache and recompute from scratch |

Exit codes are distinct so a script can tell "not in this video" from "broken":

| Code | Meaning |
|---|---|
| `0` | Found |
| `1` | No match — the closest transcript lines are printed instead |
| `2` | Error (bad URL, playlist, unsupported media, network failure) |

### Web interface

```bash
python -m uvicorn src.api:app --port 8000
```

Open <http://127.0.0.1:8000>. Progress streams live over Server-Sent Events, so
a long transcription shows a moving bar rather than a frozen page.

| Endpoint | Purpose |
|---|---|
| `POST /api/find` | Start a search → `202` + `job_id` |
| `GET /api/jobs/{id}` | Current state (polling alternative to SSE) |
| `GET /api/jobs/{id}/events` | Progress as Server-Sent Events |
| `GET /api/frame/{id}.png` | The extracted frame |
| `GET /api/health` | Liveness + confirms ffmpeg is on PATH |
| `GET /docs` | Interactive OpenAPI docs |

---

## 3. What to expect on timing

**The first query on a video is slow; every later one is instant.** Almost all
the cost is speech recognition, which runs once per video and is then cached.

| | First query on a video | Any later query |
|---|---|---|
| 19s clip | ~20s | ~0.1s |
| 13 min video | ~4 min | ~0.3s |
| 96 min film | ~12 min | ~0.3s |

Transcription runs at roughly 2–9x realtime on CPU. The spread is because
silence is skipped, so a sparse film costs far less per minute than a dense
interview. A new query on an already-indexed video costs under a second, or
about 10s if a frame has to be pulled fresh from deep in a long stream.

If you have a CUDA GPU, `DIALOGUEFRAME_ASR_DEVICE=cuda DIALOGUEFRAME_ASR_COMPUTE_TYPE=float16`
will cut the first-query cost dramatically.

---

## 4. Reproducing the results

`scripts/test_matrix.py` runs a set of searches with known-correct timestamps
under a hard per-case deadline, and prints a table. A case that overruns is
killed and reported as an overrun rather than left to run for an hour.

```bash
python -m scripts.test_matrix                 # every case, caches cleared first
python -m scripts.test_matrix --warm          # reuse whatever is cached
python -m scripts.test_matrix --limit 300     # change the deadline (default 180s)
python -m scripts.test_matrix --only okru-54m # run one case
```

The cases cover both hosts and the situations that have actually broken this
pipeline before: a multi-language upload (14 audio tracks), a feature-length
film, an ok.ru video with a real audio-only track, an ok.ru video with **no**
audio-only track, and a line that is genuinely absent — where `NO MATCH` is the
correct answer, not a failure.

Expected output with a warm cache:

```
CASE          TIME       OUTCOME   TIMESTAMP      FRAME     SCORE
------------  ---------  --------  -------------  --------  ------
yt-19s        3.4s       SUCCESS   00:00:07.380   110       82.9
yt-13m-dub    0.3s       SUCCESS   00:00:07.910   237       100.0
yt-13m-late   0.3s       SUCCESS   00:06:07.360   11009     100.0
yt-96m        0.2s       SUCCESS   00:06:07.830   11023     100.0
okru-63m      0.2s       SUCCESS   00:03:09.860   5690      100.0
okru-54m      4.0s       SUCCESS   00:05:25.120   7795      100.0
okru-absent   0.2s       NO MATCH (expected)

  7/7 SUCCESS, 0 overran the 180s limit
```

A cold run of the same suite takes roughly 30 minutes, nearly all of it
transcribing the two feature-length videos.

---

## 5. Layout

```
src/                 all pipeline source
  config.py          every tunable constant, each env-overridable
  service.py         find_dialogue() — the whole pipeline in one function
  cli.py             terminal front end
  api.py             FastAPI backend
  jobs.py            job state for the web layer
  paths.py           the on-disk cache layout
  errors.py          the error hierarchy every caller catches
  progress.py        the progress-callback contract
  core/              the pipeline stages
    resolve.py         yt-dlp metadata and stream selection
    audio.py           audio → 16kHz mono wav, plus a probe of the video stream
    asr.py             faster-whisper transcription with word timestamps
    normalize.py       the shared text folding used at index AND query time
    index.py           transcript cache: flat text + char→word offsets
    matching.py        fuzzy search over that index
    frame.py           timestamp → frame number → PNG via ranged seek
    ffmpeg.py          subprocess wrapper for ffmpeg/ffprobe
assets/              the web frontend
scripts/             check_env.py, test_matrix.py
outputs/             generated per-video artifacts (gitignored)
```

Every stage runs on its own, which is how you tell a mishearing from a match
failure:

```bash
python -m src.core.resolve   "<url>"                    # what stream did we pick?
python -m src.core.audio     "<url>"                    # fetch audio + probe video
python -m src.core.asr       outputs/<key>/audio.wav    # what did the model hear?
python -m src.core.index     <media_key>                # inspect the transcript
python -m src.core.matching  <media_key> "<text>"       # why did/didn't it match?
python -m src.core.frame     <media_key> <seconds>      # extract one frame
```

### How results are stored

```
outputs/me-at-the-zoo-103eea2ce1/
    audio.wav          16kHz mono PCM — the transcription input
    audio.meta.json    which audio track that wav came from
    probe.json         fps, dimensions, VFR flag, source URL
    transcript.json    words, timings, normalized text, char→word offsets
    index.json         manifest (version, model, counts)
    frames/frame_000007380.png
```

The directory name is a readable title slug plus a 10-character identity hash.
The slug is only a label; the hash is derived from the extractor and video id,
so two videos sharing a title cannot collide, and **a video renamed upstream
keeps its cache** instead of silently re-transcribing.

Delete a video's folder to force a full recompute, or delete `outputs/` to
start clean.

**Worked examples are committed.** `outputs/` holds the real artifacts from six
videos across both hosts, so every output format can be inspected without
running anything. Excluded from the repo: `audio.wav` (496 MB of raw PCM that
shows nothing and regenerates on demand), `_resolve/` (cached stream URLs
carrying expiry tokens), and the artifacts of two commercial videos that are
still used as test cases. All of it is recreated automatically, so a fresh
clone works unchanged.

---

## 6. Configuration

Everything lives in `src/config.py` and every value is overridable by an
environment variable. The ones worth knowing:

| Variable | Default | Effect |
|---|---|---|
| `DIALOGUEFRAME_ASR_MODEL` | `small` | `tiny`/`base`/`small`/`medium`/`large-v3`. Smaller is **not** reliably faster — see below. |
| `DIALOGUEFRAME_ASR_DEVICE` | `cpu` | Set `cuda` if you have a GPU |
| `DIALOGUEFRAME_ASR_BEAM_SIZE` | `1` | Greedy decoding; `5` trades ~3x speed for marginal accuracy |
| `DIALOGUEFRAME_MAX_DURATION` | `7200` | Refuse videos longer than this (seconds) |
| `DIALOGUEFRAME_MATCH_THRESHOLD` | `70` | Fuzzy score below which nothing is a match |
| `DIALOGUEFRAME_AUDIO_LANGUAGE` | *(original)* | Force a specific dub on a multi-language upload |
| `DIALOGUEFRAME_OUTPUT_DIR` | `outputs` | Where artifacts are written |

Two defaults are counter-intuitive and were set by measurement:

- **`beam_size=1`** is 2.95x faster than beam 5 on identical audio. Matching is
  fuzzy, so the accuracy beam search buys does not change whether a line is
  found.
- **A smaller model is not reliably faster.** `base` measured 3.5x *slower*
  than `small` on the same audio, because decode cost tracks the number of
  tokens emitted and small models fall into repetition loops.

Do not lower `DIALOGUEFRAME_MATCH_THRESHOLD` below 70 without re-testing: at 70 there
were 0 false matches in a 184-phrase sample, and 3 at 65.

---

## 7. Troubleshooting

**"No audio stream found in any available format"** — the host offered nothing
usable. Some ok.ru videos have no audio-only track; the pipeline falls back to
the smallest progressive stream and warns that it is fetching video bytes.

**A line you can hear is reported as not found.** Check what the model actually
heard before touching the threshold:

```bash
python -m src.core.asr outputs/<media_key>/audio.wav --limit 100
```

If the transcript is in the wrong language, the video probably has multiple
audio tracks and a dub was selected. The original is preferred automatically;
`DIALOGUEFRAME_AUDIO_LANGUAGE=en` forces a specific one.

**ok.ru fails with a connection reset.** ok.ru resets a share of connections
from some networks. Transport failures are retried automatically (8 attempts
with backoff); `DIALOGUEFRAME_RESOLVE_ATTEMPTS=16` buys more patience.

**Frame number is `null`.** The source is variable-frame-rate, so a frame
number cannot be derived from a timestamp by arithmetic. The extracted image is
still the correct frame — only the integer is unavailable.

---

## 8. Limitations

- The first query on a long video takes minutes. Reaching "any video in seconds"
  needs a GPU or a hosted ASR API, not more CPU tuning.
- Jobs live in the web server's memory, so they do not survive a restart and
  the API should run as a single worker.
- Digits are not spoken-form normalized: a query for `1975` will not match a
  transcript that reads "nineteen seventy five".
- Accuracy is bounded by the transcription. Heavy accents, overlapping speech,
  and music under dialogue all reduce it.
