"""
Voice I/O for the interview bot.

REQUIREMENTS:
    pip install pyttsx3 SpeechRecognition pyaudio gTTS pygame numpy

Linux note: pyaudio needs PortAudio's dev headers first:
    sudo apt-get install portaudio19-dev
    pip install pyaudio

WHAT CHANGED VS. THE FIRST VERSION
-----------------------------------
1. Manual stop: recognizer.listen() (the original approach) is a single
   blocking call with no way to interrupt it early from outside. Background
   noise (a running fan) can also throw off its internal silence detection,
   causing it to either cut you off mid-sentence or never detect silence at
   all. This version replaces that with a hand-rolled recording loop that
   reads small audio chunks (~30ms each) and checks two stop conditions on
   every chunk: (a) enough continuous silence has passed (pause_seconds),
   same idea as before, or (b) you pressed ENTER. Whichever happens first
   ends the recording — instantly for Enter, since it's checked every chunk.

2. Noise robustness: ambient calibration now runs for longer (2.5s) to get
   a more stable read on steady background noise like a fan, and the
   speech-vs-silence energy threshold is set with a margin above that
   baseline so a constant fan hum doesn't get misread as continuous speech
   (which would prevent silence from ever being detected) or as noise loud
   enough to trigger false starts. If your fan is still causing problems,
   the single most effective fix is physical, not code: move the mic
   further from the fan, or point a directional/headset mic away from it —
   no amount of thresholding fully substitutes for that with a free,
   general-purpose recognizer.

3. Indian-accented voice: pyttsx3 depends entirely on voices installed on
   the OS, and a proper Indian English voice is realistically only
   available out of the box on some Windows installs (e.g. "Microsoft
   Heera"/"Ravi"), not on Linux/Mac. For a voice that reliably sounds
   Indian-accented on any machine, this version uses gTTS (Google
   Text-to-Speech) with `tld="co.in"`, which is Google's Indian-English
   accent variant, played back via pygame. This needs internet — which you
   already require for Google speech recognition, so it's not a new
   constraint.

HONESTY NOTE ON "VOICE CONFIDENCE"
-----------------------------------
speech_recognition + Google's recognizer only returns TEXT — no pitch,
tremor, or emotion data. `voice_confidence_heuristic` below is a crude,
clearly-labeled pacing-based proxy, not a validated confidence measurement.
Real acoustic confidence detection needs a dedicated prosody/paralinguistic
ML model — a separate project on top of this one.
"""

import os
import tempfile
import threading
import time

import numpy as np
import pyttsx3
import speech_recognition as sr

try:
    import pygame
except ImportError:
    pygame = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None


# ---------------------------------------------------------------------
# Speaking
# ---------------------------------------------------------------------

_TLD_BY_ACCENT = {
    "indian": "co.in",
    "us": "com",
    "uk": "co.uk",
    "australian": "com.au",
}


def speak(text: str, accent: str = "indian", rate: int = 170):
    """
    Speak `text` out loud.

    accent="indian" (default): uses gTTS with tld="co.in" — a reliable
        Indian-English accent on any OS, played back via pygame. Requires
        internet.
    accent="us" / "uk" / "australian": other gTTS accent variants.
    accent="system": falls back to the local pyttsx3 system voice instead
        (works offline, but the accent depends entirely on what's
        installed on this machine).
    """
    if accent == "system" or gTTS is None or pygame is None:
        _speak_system_voice(text, rate)
        return

    tld = _TLD_BY_ACCENT.get(accent, "co.in")
    try:
        tts = gTTS(text=text, lang="en", tld=tld)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            temp_path = f.name
        tts.save(temp_path)

        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.quit()
        os.remove(temp_path)
    except Exception as e:
        print(f"[gTTS playback failed ({e}) — falling back to system voice]")
        _speak_system_voice(text, rate)


def _speak_system_voice(text: str, rate: int = 170):
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def list_system_voices():
    """Utility: print installed pyttsx3 voices, in case an Indian voice IS
    installed on this machine (mainly relevant on Windows) and you'd rather
    use accent='system' with a specific voice id."""
    engine = pyttsx3.init()
    for v in engine.getProperty("voices"):
        print(v.id, "-", v.name)


# ---------------------------------------------------------------------
# Listening
# ---------------------------------------------------------------------

def _rms(chunk_bytes: bytes, sample_width: int) -> float:
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
    data = np.frombuffer(chunk_bytes, dtype=dtype).astype(np.float64)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data))))


def listen_for_answer(
    pause_seconds: float = 6.5,
    timeout: float = 8.0,
    phrase_time_limit: float = 180.0,
    allow_manual_stop: bool = True,
    noise_margin: float = 1.8,
):
    """
    Listens on the default microphone for the candidate's spoken answer.
    Stops on whichever comes first:
      - `pause_seconds` of continuous silence after speech was detected
        (your "6-7 second pause" requirement), or
      - the candidate/interviewer pressing ENTER (if allow_manual_stop),
        useful when background noise (e.g. a fan) makes the automatic
        pause detection unreliable, or
      - `phrase_time_limit` as a hard safety cap, or
      - `timeout` seconds passing with no speech detected at all (treated
        as no answer).

    noise_margin: how far above the calibrated ambient noise level a chunk's
        energy must be to count as "speech" rather than background noise.
        Raise this (e.g. to 2.5-3.0) if a running fan is still being picked
        up as continuous "speech" and silence never gets detected.

    Returns:
        (text, voice_metrics)
        voice_metrics: {"duration_seconds", "words_per_minute",
                        "no_response", "stopped_by"}
        stopped_by is one of "pause", "manual_enter", "phrase_time_limit",
        or "timeout".
    """
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    stop_event = threading.Event()
    if allow_manual_stop:
        def _wait_for_enter():
            try:
                input()
            except EOFError:
                pass
            stop_event.set()

        threading.Thread(target=_wait_for_enter, daemon=True).start()
        print("(Press ENTER at any time to stop early instead of waiting for the pause.)")

    with mic as source:
        # Longer calibration so a steady fan hum is baked into the ambient
        # baseline rather than mistaken for speech.
        recognizer.adjust_for_ambient_noise(source, duration=2.5)
        speech_threshold = recognizer.energy_threshold * noise_margin

        chunk_size = source.CHUNK
        sample_rate = source.SAMPLE_RATE
        sample_width = source.SAMPLE_WIDTH
        seconds_per_chunk = chunk_size / float(sample_rate)

        frames = []
        speech_detected = False
        silence_run = 0.0
        stopped_by = "timeout"
        start_time = time.time()

        while True:
            if stop_event.is_set():
                stopped_by = "manual_enter"
                break

            chunk = source.stream.read(chunk_size)
            frames.append(chunk)

            level = _rms(chunk, sample_width)
            elapsed = time.time() - start_time

            if level >= speech_threshold:
                speech_detected = True
                silence_run = 0.0
            else:
                silence_run += seconds_per_chunk

            if not speech_detected and elapsed >= timeout:
                stopped_by = "timeout"
                break
            if speech_detected and silence_run >= pause_seconds:
                stopped_by = "pause"
                break
            if elapsed >= phrase_time_limit:
                stopped_by = "phrase_time_limit"
                break

    duration = round(time.time() - start_time, 2)

    if not speech_detected:
        return "", {
            "duration_seconds": 0.0,
            "words_per_minute": 0.0,
            "no_response": True,
            "stopped_by": stopped_by,
        }

    audio_data = sr.AudioData(b"".join(frames), sample_rate, sample_width)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = ""
    except sr.RequestError:
        text = ""

    words = len(text.split()) if text else 0
    wpm = round((words / duration) * 60, 1) if duration > 0 and words > 0 else 0.0

    voice_metrics = {
        "duration_seconds": duration,
        "words_per_minute": wpm,
        "no_response": text == "",
        "stopped_by": stopped_by,
    }
    return text, voice_metrics


def voice_confidence_heuristic(voice_metrics: dict) -> float:
    """
    Crude 0-100 pacing-based heuristic. NOT a validated confidence
    measurement — see the module docstring. Use only as auxiliary context
    alongside Agent 6's text-based confidence_score.
    """
    if voice_metrics.get("no_response"):
        return 0.0
    wpm = voice_metrics.get("words_per_minute", 0)
    if wpm == 0:
        return 0.0
    ideal_low, ideal_high = 90, 170
    if ideal_low <= wpm <= ideal_high:
        return 100.0
    distance = min(abs(wpm - ideal_low), abs(wpm - ideal_high))
    return max(0.0, round(100 - distance, 1))