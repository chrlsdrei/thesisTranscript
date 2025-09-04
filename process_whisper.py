#!/usr/bin/env python3
"""
Improved Whisper Transcription Script for Multi-Part Audio Processing
Combines best features from both versions
"""

import os
import sys
import json
import whisper
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

# Configuration
MODEL = "medium"  # Can be: tiny, base, small, medium, large, large-v2, large-v3
LANGUAGE = "tl"  # Tagalog - change as needed
MIN_SENTENCE_WORDS = 3  # Minimum words for sentence utterances
MAX_SENTENCE_WORDS = 8  # Maximum words for sentence utterances

# Folders
BASE_DIR = Path(".")
AUDIO_INPUT_DIR = BASE_DIR / "audio_input"
FULL_TRANSCRIPT_DIR = BASE_DIR / "new_full_transcript"
SEGMENTS_OUTPUT_DIR = BASE_DIR / "new_segments_output"
TEXT_OUTPUT_DIR = BASE_DIR / "new_text_output"

# Create directories if they don't exist
for dir_path in [AUDIO_INPUT_DIR, FULL_TRANSCRIPT_DIR, SEGMENTS_OUTPUT_DIR, TEXT_OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class WhisperTranscriber:
    def __init__(self, model_name=MODEL, language=LANGUAGE):
        """Initialize the Whisper model"""
        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        self.language = language
        
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio file using Whisper with optimized parameters"""
        print(f"Transcribing: {audio_path}")
        
        # Use more aggressive parameters for better segmentation
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=True,
            verbose=False,
            temperature=0,
            compression_ratio_threshold=1.8,  # More aggressive splitting
            logprob_threshold=-0.5,  # Lower threshold for more segments
            no_speech_threshold=0.3,  # Lower threshold to catch more speech
            condition_on_previous_text=False,  # Prevent context carryover
            initial_prompt=None  # No prompt to avoid bias
        )
        
        return result
    
    def split_word_segments(self, segments: List[Dict]) -> List[Dict]:
        """Split segments containing multiple words (comma-separated) into individual utterances"""
        new_segments = []
        
        for seg in segments:
            text = seg.get('text', '').strip()
            
            # Check for comma-separated words (isolated words section)
            if ',' in text and len(text.split(',')) > 2:
                words = [w.strip() for w in text.split(',') if w.strip()]
                
                if 'words' in seg and len(seg['words']) > 0:
                    # Use actual word timestamps if available
                    for word_info in seg['words']:
                        word_text = word_info.get('word', '').strip()
                        if word_text:
                            new_segments.append({
                                'start': word_info.get('start', seg['start']),
                                'end': word_info.get('end', seg['end']),
                                'text': word_text,
                                'avg_logprob': word_info.get('probability', seg.get('avg_logprob', -0.5))
                            })
                else:
                    # Fallback: distribute time evenly
                    duration = seg['end'] - seg['start']
                    word_duration = duration / len(words)
                    
                    for j, word in enumerate(words):
                        new_segments.append({
                            'start': seg['start'] + (j * word_duration),
                            'end': seg['start'] + ((j + 1) * word_duration),
                            'text': word,
                            'avg_logprob': seg.get('avg_logprob', -0.5)
                        })
            else:
                # Keep original segment
                new_segments.append(seg)
        
        return new_segments
    
    def detect_speech_parts_hybrid(self, segments: List[Dict]) -> Dict[str, List[Tuple[int, Dict]]]:
        """
        Hybrid detection using both pattern recognition and silence detection
        """
        parts = {
            "sentences": [],
            "isolated_words": [],
            "syllables": [],
            "context_words": []
        }
        
        if not segments:
            return parts
        
        # First, try pattern-based detection
        for i, seg in enumerate(segments):
            text = seg.get('text', '').strip()
            word_count = len(text.split())
            duration = seg.get('end', 0) - seg.get('start', 0)
            char_count = len(text.replace(' ', ''))
            
            # Classification based on content patterns
            if word_count >= 3 and duration > 2.0:
                # Multi-word utterances = sentences
                parts["sentences"].append((i, seg))
            elif char_count <= 4 and duration < 1.5:
                # Very short = syllables
                parts["syllables"].append((i, seg))
            elif word_count == 1 and duration < 3.0:
                # Single words
                # Use position to determine if isolated or context
                total_segments = len(segments)
                if i > total_segments * 0.7:  # Last 30% likely context words
                    parts["context_words"].append((i, seg))
                else:
                    parts["isolated_words"].append((i, seg))
            else:
                # Default assignment based on position
                total_segments = len(segments)
                if i < total_segments * 0.3:
                    parts["sentences"].append((i, seg))
                elif i < total_segments * 0.5:
                    parts["isolated_words"].append((i, seg))
                elif i < total_segments * 0.7:
                    parts["syllables"].append((i, seg))
                else:
                    parts["context_words"].append((i, seg))
        
        return parts
    
    def process_all_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Process all segments without strict part boundaries
        Ensures nothing is missed
        """
        utterances = []
        
        for i, seg in enumerate(segments):
            text = seg.get('text', '').strip()
            if not text:
                continue
                
            # Check if we need to split this segment
            if 'words' in seg and len(seg['words']) > 0:
                # Use word-level timestamps when available
                words = seg['words']
                word_count = len(words)
                
                # For multi-word segments, decide whether to keep together or split
                if word_count > MAX_SENTENCE_WORDS:
                    # Split long segments
                    current_group = []
                    current_start = None
                    
                    for word in words:
                        if current_start is None:
                            current_start = word.get('start', seg['start'])
                        
                        current_group.append(word)
                        
                        if len(current_group) >= MAX_SENTENCE_WORDS:
                            # Create utterance from group
                            utterances.append({
                                'start': current_start,
                                'end': current_group[-1].get('end', seg['end']),
                                'text': ' '.join([w.get('word', '').strip() for w in current_group]),
                                'confidence': float(np.mean([w.get('probability', 0.5) for w in current_group]))
                            })
                            current_group = []
                            current_start = None
                    
                    # Add remaining words
                    if current_group:
                        utterances.append({
                            'start': current_start,
                            'end': current_group[-1].get('end', seg['end']),
                            'text': ' '.join([w.get('word', '').strip() for w in current_group]),
                            'confidence': float(np.mean([w.get('probability', 0.5) for w in current_group]))
                        })
                elif word_count >= MIN_SENTENCE_WORDS:
                    # Keep as sentence
                    utterances.append({
                        'start': seg.get('start', 0),
                        'end': seg.get('end', 0),
                        'text': text,
                        'confidence': seg.get('avg_logprob', -0.5)
                    })
                else:
                    # Process as individual words
                    for word in words:
                        word_text = word.get('word', '').strip()
                        if word_text:
                            utterances.append({
                                'start': word.get('start', seg['start']),
                                'end': word.get('end', seg['end']),
                                'text': word_text,
                                'confidence': word.get('probability', 0.5)
                            })
            else:
                # No word timestamps, use segment as-is
                utterances.append({
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0),
                    'text': text,
                    'confidence': seg.get('avg_logprob', -0.5)
                })
        
        return utterances
    
    def process_audio_file(self, audio_path: Path) -> Tuple[List[Dict], str, str]:
        """Process a single audio file and return all utterances"""
        # Transcribe with optimized parameters
        result = self.transcribe_audio(str(audio_path))
        segments = result.get('segments', [])
        
        if not segments:
            print("⚠️ No segments found in transcription")
            return [], "", ""
        
        print(f"  Initial segments: {len(segments)}")
        
        # Apply word-level splitting for comma-separated lists
        segments = self.split_word_segments(segments)
        print(f"  After word splitting: {len(segments)}")
        
        # Process all segments (no risk of missing content)
        utterances = self.process_all_segments(segments)
        print(f"  Final utterances: {len(utterances)}")
        
        # Detect parts for analysis (but don't filter)
        parts = self.detect_speech_parts_hybrid(segments)
        
        # Show part distribution
        for part_name, part_items in parts.items():
            if part_items:
                print(f"  {part_name}: {len(part_items)} segments")
        
        # Generate Kaldi format data
        base_name = audio_path.stem
        segments_data = []
        text_data = []
        
        for i, utt in enumerate(utterances, 1):
            utt_id = f"{base_name}_{i:04d}"
            segments_data.append(
                f"{utt_id} {base_name} {utt['start']:.2f} {utt['end']:.2f}"
            )
            text_data.append(f"{utt_id} {utt['text']}")
        
        return utterances, '\n'.join(segments_data), '\n'.join(text_data)


def save_outputs(audio_path: Path, utterances: List[Dict], segments_data: str, text_data: str):
    """Save all outputs to respective directories"""
    base_name = audio_path.stem
    
    # Save full transcript with detailed info
    transcript_path = FULL_TRANSCRIPT_DIR / f"{base_name}_transcript.txt"
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcription for: {base_name}\n")
        f.write(f"Total utterances: {len(utterances)}\n")
        f.write("="*50 + "\n\n")
        
        for i, utt in enumerate(utterances, 1):
            f.write(f"Utterance {i}\n")
            f.write(f"Start: {utt['start']:.2f}\n")
            f.write(f"End: {utt['end']:.2f}\n")
            f.write(f"Duration: {utt['end'] - utt['start']:.2f}s\n")
            f.write(f"Confidence: {utt.get('confidence', -1):.4f}\n")
            f.write(f"Text: {utt['text']}\n\n")
    
    # Save segments (Kaldi format)
    segments_path = SEGMENTS_OUTPUT_DIR / f"{base_name}_segments"
    with open(segments_path, 'w', encoding='utf-8') as f:
        f.write(segments_data)
    
    # Save text (Kaldi format)
    text_path = TEXT_OUTPUT_DIR / f"{base_name}_text"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(text_data)
    
    print(f"✓ Saved outputs for {base_name}")
    print(f"  - Transcript: {transcript_path}")
    print(f"  - Segments: {segments_path}")
    print(f"  - Text: {text_path}")


def analyze_transcription_quality(utterances: List[Dict]) -> Dict:
    """Analyze the quality of transcription"""
    if not utterances:
        return {"quality": "poor", "reason": "No utterances found"}
    
    # Calculate metrics
    total_duration = sum(utt['end'] - utt['start'] for utt in utterances)
    avg_duration = total_duration / len(utterances)
    
    word_counts = [len(utt['text'].split()) for utt in utterances]
    avg_words = np.mean(word_counts)
    
    confidences = [utt.get('confidence', -1) for utt in utterances]
    avg_confidence = np.mean(confidences)
    
    # Determine quality
    quality_score = 0
    issues = []
    
    if avg_confidence > -0.5:
        quality_score += 1
    else:
        issues.append(f"Low confidence: {avg_confidence:.3f}")
    
    if avg_duration > 1.0:
        quality_score += 1
    else:
        issues.append(f"Short utterances: {avg_duration:.2f}s avg")
    
    if len(utterances) > 10:
        quality_score += 1
    else:
        issues.append(f"Few utterances: {len(utterances)}")
    
    # Determine overall quality
    if quality_score == 3:
        quality = "excellent"
    elif quality_score == 2:
        quality = "good"
    elif quality_score == 1:
        quality = "fair"
    else:
        quality = "poor"
    
    return {
        "quality": quality,
        "total_utterances": len(utterances),
        "avg_duration": avg_duration,
        "avg_words": avg_words,
        "avg_confidence": avg_confidence,
        "total_duration": total_duration,
        "issues": issues
    }


def get_wav_files() -> List[Path]:
    """Get list of WAV files in input directory"""
    return sorted(AUDIO_INPUT_DIR.glob("*.wav"))


def display_menu(wav_files: List[Path]):
    """Display file selection menu"""
    print("\n" + "="*50)
    print("IMPROVED WHISPER TRANSCRIPTION TOOL")
    print("="*50)
    print(f"Model: {MODEL} | Language: {LANGUAGE}")
    print(f"Found {len(wav_files)} WAV files:\n")
    
    for i, file in enumerate(wav_files, 1):
        print(f"  {i}. {file.name}")
    
    print("\nOptions:")
    print("  - Enter a number to process single file")
    print("  - Enter range (e.g., '1-5') to process multiple files")
    print("  - Enter 'all' to process all files")
    print("  - Enter 'q' to quit")
    print("-"*50)


def parse_selection(selection: str, max_files: int) -> List[int]:
    """Parse user selection and return list of file indices"""
    selection = selection.strip().lower()
    
    if selection == 'q':
        return []
    
    if selection == 'all':
        return list(range(max_files))
    
    # Check for range
    if '-' in selection:
        try:
            start, end = selection.split('-')
            start = int(start.strip()) - 1
            end = int(end.strip())
            return list(range(start, end))
        except:
            print("Invalid range format. Use format like '1-5'")
            return []
    
    # Multiple comma-separated
    if ',' in selection:
        try:
            indices = []
            for num in selection.split(','):
                idx = int(num.strip()) - 1
                if 0 <= idx < max_files:
                    indices.append(idx)
            return indices
        except:
            print("Invalid format")
            return []
    
    # Single number
    try:
        idx = int(selection) - 1
        if 0 <= idx < max_files:
            return [idx]
        else:
            print(f"Number must be between 1 and {max_files}")
            return []
    except:
        print("Invalid input")
        return []


def main():
    """Main execution function"""
    print("Initializing Improved Whisper Transcription System...")
    
    # Get WAV files
    wav_files = get_wav_files()
    
    if not wav_files:
        print(f"No WAV files found in {AUDIO_INPUT_DIR}")
        sys.exit(1)
    
    # Initialize transcriber
    transcriber = WhisperTranscriber(model_name=MODEL, language=LANGUAGE)
    
    while True:
        display_menu(wav_files)
        selection = input("\nEnter your selection: ")
        
        if selection.lower() == 'q':
            print("Exiting...")
            break
        
        indices = parse_selection(selection, len(wav_files))
        
        if not indices:
            continue
        
        # Process selected files
        print(f"\nProcessing {len(indices)} file(s)...")
        print("-"*50)
        
        results_summary = []
        
        for idx in indices:
            audio_path = wav_files[idx]
            print(f"\n📊 Processing: {audio_path.name}")
            
            try:
                utterances, segments_data, text_data = transcriber.process_audio_file(audio_path)
                
                if utterances:
                    save_outputs(audio_path, utterances, segments_data, text_data)
                    
                    # Analyze quality
                    analysis = analyze_transcription_quality(utterances)
                    
                    results_summary.append({
                        'file': audio_path.name,
                        'utterances': len(utterances),
                        'quality': analysis['quality'],
                        'duration': analysis['total_duration']
                    })
                    
                    print(f"  ✓ Quality: {analysis['quality'].upper()}")
                    print(f"  ✓ Total duration: {analysis['total_duration']:.1f}s")
                    
                    if analysis['issues']:
                        print(f"  ⚠️ Issues: {', '.join(analysis['issues'])}")
                else:
                    print(f"  ✗ No utterances generated")
                    results_summary.append({
                        'file': audio_path.name,
                        'utterances': 0,
                        'quality': 'failed',
                        'duration': 0
                    })
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                results_summary.append({
                    'file': audio_path.name,
                    'error': str(e)
                })
        
        # Show summary if multiple files
        if len(results_summary) > 1:
            print("\n" + "="*50)
            print("BATCH PROCESSING SUMMARY")
            print("="*50)
            
            for result in results_summary:
                if 'error' in result:
                    print(f"❌ {result['file']}: ERROR - {result['error']}")
                elif result['utterances'] == 0:
                    print(f"⚠️ {result['file']}: No utterances")
                else:
                    print(f"✅ {result['file']}: {result['utterances']} utterances, "
                          f"Quality: {result['quality'].upper()}, "
                          f"Duration: {result['duration']:.1f}s")
        
        another = input("\nProcess more files? (y/n): ")
        if another.lower() != 'y':
            break
    
    print("\nThank you for using the Improved Whisper Transcription Tool!")


if __name__ == "__main__":
    main()