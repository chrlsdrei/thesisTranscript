#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process_whisper.py

Transcribe .wav files from thesisTranscript/audio_input using Whisper (prefers faster-whisper).
Outputs:
  - new_full_transcript/<basename>_full_transcript.txt
  - new_segments_output/<basename>_segments
  - new_text_output/<basename>_text

Features:
  - Interactive file picker (single, list, ranges e.g. 1-5)
  - Select Whisper model and language (default: medium, tl)
  - Optional manual boundaries for 4 parts; otherwise auto-detect via silence gaps
  - Part 1: sentence-level utterances with min/max words per sentence
  - Parts 2 & 4: word-by-word utterances
  - Part 3: syllable-by-syllable utterances (treated as word tokens from ASR)
  - Kaldi outputs follow: <uttid> <recording-id> <start> <end>
    where uttid = <basename>_<4-digit seq>

Notes:
  - Confidence reported as average log probability if available; else a safe placeholder (None)
  - Requires ffmpeg in PATH. If available, pydub silence detection improves part splitting.
"""

import os
import re
import sys
import math
import glob
import json
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional

AUDIO_ROOT = os.path.join(".", "audio_input")
FULL_OUT = os.path.join(".", "new_full_transcript")
SEG_OUT = os.path.join(".", "new_segments_output")
TEXT_OUT = os.path.join(".", "new_text_output") 

# -------- Adjustable defaults --------
DEFAULT_MODEL = "medium"
DEFAULT_LANGUAGE = "tl"   # e.g. "tl" (Tagalog), "en", etc.

# For Part 1 sentence grouping
SENT_MIN_WORDS = 3   # you can raise (e.g., 3) if you want to merge very short sentences
SENT_MAX_WORDS = 10  # upper cap to avoid run-ons

# Silence detection for auto 4-part split
# We'll consider 'long silences' that likely separate the parts
LONG_SILENCE_MS = 1500  # gap >= 1.5s considered a candidate boundary
TOP_GAPS_TO_PICK = 3    # we need 3 boundaries to produce 4 parts

# Fallback if pydub is unavailable: we’ll derive gaps from ASR segment boundaries
USE_PYDUB_IF_AVAILABLE = True

# -------------------------------------

# Try imports conditionally
HAVE_PYDUB = False
if USE_PYDUB_IF_AVAILABLE:
    try:
        from pydub import AudioSegment, silence
        HAVE_PYDUB = True
    except Exception:
        HAVE_PYDUB = False

HAVE_FASTER = False
try:
    from faster_whisper import WhisperModel as FWModel  # type: ignore
    HAVE_FASTER = True
except Exception:
    HAVE_FASTER = False

HAVE_OPENAI_WHISPER = False
try:
    import whisper  # openai/whisper
    HAVE_OPENAI_WHISPER = True
except Exception:
    HAVE_OPENAI_WHISPER = False

if not HAVE_FASTER and not HAVE_OPENAI_WHISPER:
    print("ERROR: Neither faster-whisper nor openai-whisper is available. Please install one of them.")
    sys.exit(1)

@dataclass
class WordTok:
    text: str
    start: float
    end: float
    avg_logprob: Optional[float] = None

@dataclass
class SegTok:
    text: str
    start: float
    end: float
    avg_logprob: Optional[float] = None
    words: Optional[List[WordTok]] = None  # may be None if not available


def list_audio_files() -> List[str]:
    files = sorted(glob.glob(os.path.join(AUDIO_ROOT, "*.wav")))
    return files


def parse_selection(inp: str, n: int) -> List[int]:
    """
    Parse user input like "1", "1,3,7", "1-5", "1-3,7,10-12" into zero-based indices.
    """
    idxs = set()
    tokens = [t.strip() for t in inp.split(",") if t.strip()]
    for tok in tokens:
        if "-" in tok:
            a, b = tok.split("-", 1)
            try:
                start = int(a) - 1
                end = int(b) - 1
                for i in range(start, end + 1):
                    if 0 <= i < n:
                        idxs.add(i)
            except ValueError:
                pass
        else:
            try:
                i = int(tok) - 1
                if 0 <= i < n:
                    idxs.add(i)
            except ValueError:
                pass
    return sorted(idxs)


def seconds_to_str(t: float) -> str:
    return f"{t:.2f}"


def sanitize_text(s: str) -> str:
    # Collapse whitespace; preserve punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def group_sentences_by_words(words: List[WordTok],
                             min_words: int,
                             max_words: int) -> List[SegTok]:
    """
    Build sentence-like chunks from tokenized words, respecting punctuation
    and min/max word constraints. Uses word timestamps to assign precise times.
    """
    utterances: List[SegTok] = []
    buf: List[WordTok] = []
    def flush():
        if not buf:
            return
        text = sanitize_text(" ".join(w.text for w in buf))
        start = buf[0].start
        end = buf[-1].end
        avg_lp = sum([w.avg_logprob for w in buf if w.avg_logprob is not None]) / max(
            1, sum(1 for w in buf if w.avg_logprob is not None)
        )
        utterances.append(SegTok(text=text, start=start, end=end, avg_logprob=avg_lp, words=list(buf)))

    for w in words:
        buf.append(w)
        end_of_sentence = bool(re.search(r"[\.!?…]+$", w.text))
        # If we hit EOS and buffer is big enough, flush
        if end_of_sentence and len(buf) >= min_words:
            flush()
            buf = []
        else:
            # If we exceed max words, flush early
            if len(buf) >= max_words:
                flush()
                buf = []

    # tail
    if buf:
        flush()
    return utterances


def evenly_split_segment_to_words(seg: SegTok) -> List[WordTok]:
    """
    Fallback when word timestamps are not available:
    split text by whitespace and distribute time evenly.
    """
    toks = [t for t in re.split(r"\s+", seg.text.strip()) if t]
    if not toks:
        return []
    duration = max(0.0, seg.end - seg.start)
    per = duration / len(toks) if len(toks) > 0 else duration
    words = []
    for i, tok in enumerate(toks):
        start = seg.start + i * per
        end = seg.start + (i + 1) * per if i < len(toks) - 1 else seg.end
        words.append(WordTok(text=tok, start=start, end=end, avg_logprob=seg.avg_logprob))
    return words


def try_pydub_boundaries(wav_path: str, total_dur_s: float) -> Optional[List[float]]:
    """
    Use pydub to find long silences and pick the 3 largest gaps as boundaries.
    Returns sorted boundary times [b1, b2, b3] or None.
    """
    if not HAVE_PYDUB:
        return None
    try:
        audio = AudioSegment.from_file(wav_path)
        # detect_silence returns list of [start_ms, end_ms]
        sils = silence.detect_silence(audio, min_silence_len=LONG_SILENCE_MS, silence_thresh=audio.dBFS - 16)
        if not sils:
            return None
        # Use silence *starts* as gap markers; pick three with widest spans first
        # score gaps by duration
        sils_sorted = sorted(sils, key=lambda ab: (ab[1] - ab[0]), reverse=True)
        boundaries = []
        for st, en in sils_sorted:
            mid = (st + en) / 2.0
            t = mid / 1000.0
            boundaries.append(t)
            if len(boundaries) == TOP_GAPS_TO_PICK:
                break
        if len(boundaries) < 3:
            return None
        boundaries = sorted(boundaries)
        # Clamp within audio
        boundaries = [max(0.0, min(total_dur_s, b)) for b in boundaries]
        return boundaries
    except Exception:
        return None


def derive_boundaries_from_segments(segments: List[SegTok], total_dur_s: float) -> Optional[List[float]]:
    """
    Fallback: derive top 3 largest gaps between ASR segments as boundaries.
    """
    if not segments:
        return None
    gaps = []
    for a, b in zip(segments[:-1], segments[1:]):
        gap = b.start - a.end
        gaps.append((gap, (a.end + b.start) / 2.0))
    if not gaps:
        return None
    gaps_sorted = sorted(gaps, key=lambda x: x[0], reverse=True)
    picks = [t for _, t in gaps_sorted[:TOP_GAPS_TO_PICK]]
    picks = sorted(picks)
    picks = [max(0.0, min(total_dur_s, t)) for t in picks]
    return picks if len(picks) == 3 else None


def split_into_parts(boundaries: List[float], total_dur_s: float) -> List[Tuple[float, float]]:
    """
    boundaries: exactly 3 sorted times -> returns 4 (start,end) windows
    """
    b1, b2, b3 = boundaries
    parts = [(0.0, b1), (b1, b2), (b2, b3), (b3, total_dur_s)]
    return parts


def load_and_transcribe(wav_path: str, model_name: str, language: str) -> Tuple[List[SegTok], float]:
    """
    Returns (segments, total_duration_seconds).
    Prefers faster-whisper (word timestamps robust), fallback to openai/whisper.
    """
    if HAVE_FASTER:
        # faster-whisper path
        model = FWModel(model_name, compute_type="auto")
        segments_iter, info = model.transcribe(wav_path, language=language, word_timestamps=True, vad_filter=True)
        total_dur = info.duration if hasattr(info, "duration") else None
        segs: List[SegTok] = []
        for seg in segments_iter:
            words = []
            avg_lp = getattr(seg, "avg_logprob", None)
            if getattr(seg, "words", None):
                for w in seg.words:
                    words.append(WordTok(text=w.word.strip(), start=float(w.start), end=float(w.end),
                                         avg_logprob=getattr(w, "prob", None)))
            segs.append(SegTok(text=sanitize_text(seg.text), start=float(seg.start), end=float(seg.end),
                               avg_logprob=avg_lp, words=words if words else None))
        # duration fallback
        if total_dur is None and segs:
            total_dur = max(segs[-1].end, 0.0)
        return segs, float(total_dur or 0.0)

    # openai/whisper fallback
    assert HAVE_OPENAI_WHISPER
    model = whisper.load_model(model_name)
    # Note: word timestamps may not be available in vanilla whisper; we still request timestamps=True
    result = model.transcribe(wav_path, language=language, verbose=False)
    segs: List[SegTok] = []
    for s in result.get("segments", []):
        text = sanitize_text(s.get("text", ""))
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start))
        avg_lp = s.get("avg_logprob", None)
        words = None
        if "words" in s and s["words"]:
            words = []
            for w in s["words"]:
                words.append(WordTok(text=w["word"].strip(), start=float(w["start"]), end=float(w["end"]),
                                     avg_logprob=w.get("prob", None)))
        segs.append(SegTok(text=text, start=start, end=end, avg_logprob=avg_lp, words=words))
    total_dur = float(result.get("duration", segs[-1].end if segs else 0.0))
    return segs, total_dur


def collect_all_words(segments: List[SegTok]) -> List[WordTok]:
    words: List[WordTok] = []
    for seg in segments:
        if seg.words:
            words.extend(seg.words)
        else:
            # evenly split if needed
            words.extend(evenly_split_segment_to_words(seg))
    # Remove empty tokens
    words = [w for w in words if w.text]
    return words


def filter_words_by_window(words: List[WordTok], start: float, end: float) -> List[WordTok]:
    return [w for w in words if (w.start >= start and w.end <= end)]


def build_utterances_for_parts(words: List[WordTok],
                               parts: List[Tuple[float, float]],
                               sent_min: int,
                               sent_max: int) -> List[SegTok]:
    """
    Generate utterances for the 4 parts:
      1) sentences (grouped by punctuation & min/max words)
      2) words (one per utterance)
      3) syllables (treated as 'words' from ASR)
      4) words (one per utterance)
    """
    out: List[SegTok] = []

    # Part 1: sentence-like
    p1_words = filter_words_by_window(words, parts[0][0], parts[0][1])
    out.extend(group_sentences_by_words(p1_words, sent_min, sent_max))

    # Part 2: word-by-word
    p2_words = filter_words_by_window(words, parts[1][0], parts[1][1])
    for w in p2_words:
        out.append(SegTok(text=sanitize_text(w.text), start=w.start, end=w.end, avg_logprob=w.avg_logprob, words=[w]))

    # Part 3: syllable-by-syllable (treated as word tokens, since this part is recorded with syllables)
    p3_words = filter_words_by_window(words, parts[2][0], parts[2][1])
    for w in p3_words:
        out.append(SegTok(text=sanitize_text(w.text), start=w.start, end=w.end, avg_logprob=w.avg_logprob, words=[w]))

    # Part 4: word-by-word
    p4_words = filter_words_by_window(words, parts[3][0], parts[3][1])
    for w in p4_words:
        out.append(SegTok(text=sanitize_text(w.text), start=w.start, end=w.end, avg_logprob=w.avg_logprob, words=[w]))

    return out


def write_full_transcript(basename: str, utterances: List[SegTok]) -> None:
    os.makedirs(FULL_OUT, exist_ok=True)
    out_path = os.path.join(FULL_OUT, f"{basename}_full_transcript.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for i, u in enumerate(utterances, start=1):
            f.write(f"Utterance {i}\n")
            f.write(f"Start: {seconds_to_str(u.start)}\n")
            f.write(f"End: {seconds_to_str(u.end)}\n")
            conf = u.avg_logprob if u.avg_logprob is not None else float("nan")
            # Keep original sign if it’s already a log prob; if faster-whisper gave prob [0..1], convert to log
            if conf is not None and conf > 0 and conf <= 1.0:
                try:
                    conf = math.log(conf)
                except Exception:
                    pass
            f.write(f"Confidence: {conf if conf is not None else 'None'}\n")
            f.write(f"Text: {u.text}\n\n")
    print(f"Saved: {out_path}")


def write_kaldi_outputs(basename: str, recording_id: str, utterances: List[SegTok]) -> None:
    os.makedirs(SEG_OUT, exist_ok=True)
    os.makedirs(TEXT_OUT, exist_ok=True)
    seg_path = os.path.join(SEG_OUT, f"{basename}_segments")
    txt_path = os.path.join(TEXT_OUT, f"{basename}_text")

    with open(seg_path, "w", encoding="utf-8") as segf, open(txt_path, "w", encoding="utf-8") as txtf:
        for i, u in enumerate(utterances, start=1):
            uttid = f"{basename}_{i:04d}"
            segf.write(f"{uttid} {recording_id} {seconds_to_str(u.start)} {seconds_to_str(u.end)}\n")
            txtf.write(f"{uttid} {u.text}\n")

    print(f"Saved: {seg_path}")
    print(f"Saved: {txt_path}")


def prompt_for_boundaries(total_dur: float, default_boundaries: Optional[List[float]]) -> List[float]:
    print("\n--- Part Boundaries ---")
    print("Your audio contains 4 parts:")
    print("  1) Sentences/paragraphs")
    print("  2) Isolated words")
    print("  3) Isolated syllables")
    print("  4) Isolated words (from Part 1)")
    print("Enter three boundary times (in seconds) to split into 4 parts (press Enter to accept auto):")

    if default_boundaries:
        print(f"Auto-detected boundaries (sec): {', '.join(f'{b:.2f}' for b in default_boundaries)}")

    user_inp = input("Manual boundaries (e.g., 600, 1500, 2100) or leave blank: ").strip()
    if not user_inp:
        if default_boundaries and len(default_boundaries) == 3:
            return default_boundaries
        else:
            # If no auto available, do equal quarters
            q = total_dur / 4.0
            return [q, 2*q, 3*q]

    try:
        parts = [float(x) for x in user_inp.split(",")]
        if len(parts) != 3:
            raise ValueError
        parts = sorted([max(0.0, min(total_dur, p)) for p in parts])
        return parts
    except Exception:
        print("Invalid input. Falling back to auto/equal quarters.")
        if default_boundaries and len(default_boundaries) == 3:
            return default_boundaries
        q = total_dur / 4.0
        return [q, 2*q, 3*q]


def main():
    print("=== Whisper Transcription Utility ===")
    print(f"Audio folder: {AUDIO_ROOT}\n")

    files = list_audio_files()
    if not files:
        print("No .wav files found in audio_input.")
        sys.exit(0)

    for i, fp in enumerate(files, start=1):
        print(f"{i}. {os.path.basename(fp)}")

    sel = input("\nSelect files to transcribe (e.g., 2 or 1,3,5-7): ").strip()
    indices = parse_selection(sel, len(files))
    if not indices:
        print("No valid selection. Exiting.")
        sys.exit(0)

    model_name = input(f"Model [{DEFAULT_MODEL}]: ").strip() or DEFAULT_MODEL
    language = input(f"Language code [{DEFAULT_LANGUAGE}]: ").strip() or DEFAULT_LANGUAGE

    # Sentence constraints for Part 1
    try:
        minw_in = input(f"Part1 min words per sentence [{SENT_MIN_WORDS}]: ").strip()
        maxw_in = input(f"Part1 max words per sentence [{SENT_MAX_WORDS}]: ").strip()
        minw = int(minw_in) if minw_in else SENT_MIN_WORDS
        maxw = int(maxw_in) if maxw_in else SENT_MAX_WORDS
        if minw < 1: minw = 1
        if maxw < minw: maxw = max(minw, SENT_MAX_WORDS)
    except Exception:
        minw, maxw = SENT_MIN_WORDS, SENT_MAX_WORDS

    print("\nStarting transcription...\n")

    for idx in indices:
        wav_path = files[idx]
        fname = os.path.basename(wav_path)
        base = os.path.splitext(fname)[0]

        # recording-id for Kaldi should typically be the file stem
        recording_id = base

        print(f"\n--- File: {fname} ---")
        segments, total_dur = load_and_transcribe(wav_path, model_name, language)

        # Prepare default boundaries via pydub or segment gaps
        default_bounds = try_pydub_boundaries(wav_path, total_dur)
        if not default_bounds:
            default_bounds = derive_boundaries_from_segments(segments, total_dur)

        # Ask user to accept/override
        boundaries = prompt_for_boundaries(total_dur, default_bounds)
        parts = split_into_parts(boundaries, total_dur)

        # Ensure we have word tokens (or evenly-split fallback)
        words = collect_all_words(segments)

        # Build utterances for all 4 parts
        utterances = build_utterances_for_parts(words, parts, minw, maxw)

        # Write outputs
        write_full_transcript(base, utterances)
        write_kaldi_outputs(base, recording_id, utterances)

    print("\nAll done.\n")


if __name__ == "__main__":
    main()
