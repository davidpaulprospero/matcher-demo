================================================================================
03 - MATCHER DEMO
================================================================================
STATUS:   not started
STACK:    Python (Streamlit or Gradio) wrapping the existing matcher pipeline
DEPLOY:   Render  (free, no card, 5GB/mo, SLEEPS AFTER 15 MIN IDLE)
          Koyeb as alternate if the sleep is a problem
PRIORITY: DAY 4


--------------------------------------------------------------------------------
THIS IS THE SITE THAT MAYS YOU STAND OUT
--------------------------------------------------------------------------------
A recruiter who clicks a WordPress theme sees a theme.

A recruiter who uploads a voiceover file and watches the pipeline match stock
footage to it sees someone who built a five-stage pipeline with:
  - embedding-based semantic search
  - LLM matching with tiered scoring
  - checkpointed state and resume-after-failure
  - provider abstraction across Gemini / Anthropic / Ollama
  - yt-dlp download orchestration
  - faster-whisper transcription
  - OTIO / EDL / XML timeline export

None of the other applicants will have this. This is the site to lead with in
the portfolio hub.


--------------------------------------------------------------------------------
SCOPE - KEEP IT SMALL
--------------------------------------------------------------------------------
The full pipeline is heavy (it downloads video, runs embeddings, calls LLM
APIs). A free-tier demo cannot do all that on a public URL.

Realistic scope for a hosted demo:
  - Accept a .srt or .txt voiceover
  - Parse the segments
  - Run the matching/caption logic against a small prebuilt candidate set
  - Show the timeline / match table

If full video download is too heavy for free hosting, that is fine. Demonstrate
the MATCHING, not the downloading. The matching is the interesting part
anyway.

Alternate approach if hosting struggles: pre-render a recorded demo and host
the video/animated GIF statically. Less impressive, but never 500s on a
recruiter's click.


--------------------------------------------------------------------------------
STEPS
--------------------------------------------------------------------------------
  1. Add a Streamlit (or Gradio) UI layer to the existing matcher repo
  2. Strip it to a path that runs without downloading full stock video
  3. requirements.txt with pinned deps
  4. Push to GitHub, deploy to Render as a Python web service
  5. *** TEST ON MOBILE DATA, NOT LOCALHOST ***
     Render free sleeps after 15 min idle. First paint cold can be 30-60s.
     A recruiter clicking a link that looks broken will not wait.

Step 5 is the one people skip. Do not skip it.


--------------------------------------------------------------------------------
SECURITY WARNING - READ BEFORE PUBLISHING
--------------------------------------------------------------------------------
This pipeline holds LLM API KEYS and calls paid APIs (Gemini/Anthropic).

Before putting ANY of this on a public URL:
  - Move every secret to environment variables. Confirm no key is in the repo.
  - Put a cap on requests. A public URL with an unpaid LLM key behind it is an
    open tab for anyone on the internet.
  - Consider requiring a shared access code or putting it behind a waitlist
    instead of public.
  - Check .gitignore covers .env

An open, unmetered LLM endpoint is a real cost. Do this first.


--------------------------------------------------------------------------------
PER-FOLDER CONVENTION
--------------------------------------------------------------------------------
When this is done, this folder needs:
  - the deployed live URL in this README
  - a 2-3 sentence blurb, copy-pasteable into a cover letter
  - a note on what the demo does NOT do (so nobody is surprised live)


--------------------------------------------------------------------------------
THE HONEST BOUNDARY
--------------------------------------------------------------------------------
This one is genuinely yours - the matcher pipeline is real work. Claim it
straight. But:

  - Never imply a client or revenue.
  - Describe the demo's real scope, not the pipeline's full capability.

Framing that works:
    "No client work yet - these are problems I picked to solve."


================================================================================