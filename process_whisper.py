import os
import json
import subprocess
import statistics

# Paths
AUDIO_DIR = "audio_input"
TRANSCRIPT_DIR = "transcript_output"
SEGMENTS_DIR = "segments_output"
TEXT_DIR = "text_output"

# Make sure output dirs exist
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(SEGMENTS_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# Whisper model
MODEL = "medium"  # can change to "small" or "large-v2"

# Quality thresholds for utterance analysis
MIN_CONFIDENCE = -0.3  # Minimum average confidence score
MIN_AVG_DURATION = 2.0  # Minimum average utterance duration in seconds
MIN_AVG_WORDS = 3  # Minimum average words per utterance
MAX_SINGLE_WORD_RATIO = 0.3  # Maximum ratio of single-word utterances

def detect_speech_parts(segments):
    """Detect the 4 parts of the structured speech recording."""
    parts = {
        "paragraphs": [],       # Part 1: Connected speech (paragraphs/sentences)
        "isolated_words": [],   # Part 2: Isolated words
        "syllables": [],        # Part 3: Isolated syllables  
        "repeated_words": []    # Part 4: Repeated words from part 1
    }
    
    current_part = "paragraphs"
    part_transitions = []
    
    # Define pattern thresholds
    PARAGRAPH_MIN_WORDS = 3
    PARAGRAPH_MIN_DURATION = 2.0
    WORD_MAX_DURATION = 2.5
    SYLLABLE_MAX_DURATION = 1.5
    SYLLABLE_MAX_CHARS = 4
    
    for i, seg in enumerate(segments):
        text = seg['text'].strip()
        word_count = len(text.split())
        duration = seg['end'] - seg['start']
        char_count = len(text.replace(' ', ''))
        
        # Detect transitions based on patterns
        if current_part == "paragraphs":
            # Look for transition to isolated words
            # Check if we start getting short, single-word utterances consistently
            if i < len(segments) - 5:  # Make sure we have enough segments to check
                upcoming_pattern = []
                for j in range(i, min(i + 5, len(segments))):
                    next_seg = segments[j]
                    next_text = next_seg['text'].strip()
                    next_words = len(next_text.split())
                    next_duration = next_seg['end'] - next_seg['start']
                    upcoming_pattern.append({
                        'words': next_words,
                        'duration': next_duration,
                        'is_short_single': next_words <= 2 and next_duration <= WORD_MAX_DURATION
                    })
                
                # If most upcoming segments are short and single words
                short_single_count = sum(1 for p in upcoming_pattern if p['is_short_single'])
                if short_single_count >= 3:  # At least 3 out of 5 are short single words
                    current_part = "isolated_words"
                    part_transitions.append(("paragraphs_to_words", i))
        
        elif current_part == "isolated_words":
            # Look for transition to syllables
            # Syllables are very short, often single characters or short sounds
            if i < len(segments) - 3:
                upcoming_syllables = []
                for j in range(i, min(i + 3, len(segments))):
                    next_seg = segments[j]
                    next_text = next_seg['text'].strip()
                    next_duration = next_seg['end'] - next_seg['start']
                    next_chars = len(next_text.replace(' ', ''))
                    is_syllable_like = (next_duration <= SYLLABLE_MAX_DURATION and 
                                       next_chars <= SYLLABLE_MAX_CHARS)
                    upcoming_syllables.append(is_syllable_like)
                
                if sum(upcoming_syllables) >= 2:  # At least 2 out of 3 are syllable-like
                    current_part = "syllables"
                    part_transitions.append(("words_to_syllables", i))
        
        elif current_part == "syllables":
            # Look for transition to repeated words
            # These would be longer than syllables, similar to isolated words
            if (duration > SYLLABLE_MAX_DURATION or 
                word_count > 1 or 
                char_count > SYLLABLE_MAX_CHARS):
                current_part = "repeated_words"
                part_transitions.append(("syllables_to_repeated", i))
        
        # Assign segment to current part
        parts[current_part].append((i, seg))
    
    return parts, part_transitions

def analyze_transcription_quality(json_file):
    """Analyze the quality of transcription and detect speech parts."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    segments = data["segments"]
    if not segments:
        return {"quality": "poor", "reason": "No segments found"}
    
    # Detect speech parts
    parts, transitions = detect_speech_parts(segments)
    
    # Calculate metrics for each part
    part_metrics = {}
    for part_name, part_segments in parts.items():
        if not part_segments:
            part_metrics[part_name] = {
                "count": 0,
                "avg_confidence": 0,
                "avg_duration": 0,
                "avg_words": 0
            }
            continue
            
        confidences = [seg.get('avg_logprob', -1.0) for _, seg in part_segments]
        durations = [seg['end'] - seg['start'] for _, seg in part_segments]
        word_counts = [len(seg['text'].strip().split()) for _, seg in part_segments]
        
        part_metrics[part_name] = {
            "count": len(part_segments),
            "avg_confidence": statistics.mean(confidences),
            "avg_duration": statistics.mean(durations),
            "avg_words": statistics.mean(word_counts),
            "segments": [(i, seg['text'].strip()) for i, seg in part_segments[:5]]  # First 5 examples
        }
    
    # Overall quality assessment (focus on paragraphs part for training)
    paragraph_metrics = part_metrics.get("paragraphs", {})
    
    quality_issues = []
    quality = "unknown"
    
    if paragraph_metrics.get("count", 0) == 0:
        quality = "poor"
        quality_issues.append("No paragraph/sentence segments found")
    else:
        p_conf = paragraph_metrics["avg_confidence"]
        p_dur = paragraph_metrics["avg_duration"]
        p_words = paragraph_metrics["avg_words"]
        
        if p_conf < -0.5:
            quality_issues.append(f"Low confidence in paragraphs ({p_conf:.3f})")
        
        if p_dur < 2.0:
            quality_issues.append(f"Short paragraph utterances ({p_dur:.1f}s)")
        
        if p_words < 3:
            quality_issues.append(f"Few words in paragraphs ({p_words:.1f})")
        
        # Determine overall quality based on paragraph quality
        if not quality_issues:
            quality = "excellent"
        elif len(quality_issues) == 1:
            quality = "good"
        elif len(quality_issues) == 2:
            quality = "fair"
        else:
            quality = "poor"
    
    return {
        "quality": quality,
        "parts": part_metrics,
        "transitions": transitions,
        "total_segments": len(segments),
        "issues": quality_issues
    }

def transcribe_audio(audio_file):
    """Run Whisper on audio and return JSON path."""
    json_out = os.path.join(TRANSCRIPT_DIR, os.path.splitext(audio_file)[0] + ".json")
    subprocess.run([
        "whisper", os.path.join(AUDIO_DIR, audio_file),
        "--model", MODEL,
        "--language", "tl",
        "--task", "transcribe",
        "--output_format", "json",
        "--output_dir", TRANSCRIPT_DIR
    ])
    return json_out

def generate_outputs(json_file):
    """Generate full transcription, segments, and text files."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_name = os.path.splitext(os.path.basename(json_file))[0]
    
    # Detect speech parts for analysis (but don't generate separate files)
    parts, transitions = detect_speech_parts(data["segments"])

    # Generate full transcription file
    full_trans_path = os.path.join(TRANSCRIPT_DIR, base_name + "_full_transcription.txt")
    with open(full_trans_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(data["segments"], start=1):
            f.write(f"Utterance {i}\n")
            f.write(f"Start: {seg['start']:.2f}\n")
            f.write(f"End: {seg['end']:.2f}\n")
            f.write(f"Confidence: {seg.get('avg_logprob', 0):.4f}\n")
            f.write(f"Text: {seg['text'].strip()}\n\n")

    # Generate segments file (Kaldi style)
    segments_path = os.path.join(SEGMENTS_DIR, base_name + "_segments.txt")
    with open(segments_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(data["segments"], start=1):
            seg_id = f"{base_name}_{i:04d}"
            f.write(f"{seg_id} {base_name} {seg['start']:.2f} {seg['end']:.2f}\n")

    # Generate text file (Kaldi style)
    text_path = os.path.join(TEXT_DIR, base_name + "_text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(data["segments"], start=1):
            seg_id = f"{base_name}_{i:04d}"
            f.write(f"{seg_id} {seg['text'].strip()}\n")

    print(f"✅ Outputs generated for {base_name}")
    print(f" - Full transcription: {full_trans_path}")
    print(f" - Segments: {segments_path}")
    print(f" - Text: {text_path}")
    
    # Show speech parts analysis summary
    parts_summary = []
    for part_name, part_segments in parts.items():
        if part_segments:
            parts_summary.append(f"{part_name.replace('_', ' ').title()}: {len(part_segments)} segments")
    
    if parts_summary:
        print(f" - Speech parts detected: {', '.join(parts_summary)}")
    
    return {
        "full_transcription": full_trans_path,
        "segments": segments_path,
        "text": text_path,
        "parts_analysis": parts
    }

def main():
    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]

    if not wav_files:
        print("⚠️ No .wav files found in audio_input/")
        return

    print("\n=== AUDIO TRANSCRIPTION AND QUALITY ANALYSIS ===")
    print("\nAnalyzing existing transcriptions...")
    
    # Check for existing transcriptions and analyze quality
    existing_analyses = {}
    for audio_file in wav_files:
        base_name = os.path.splitext(audio_file)[0]
        json_file = os.path.join(TRANSCRIPT_DIR, base_name + ".json")
        
        if os.path.exists(json_file):
            try:
                analysis = analyze_transcription_quality(json_file)
                existing_analyses[audio_file] = analysis
            except Exception as e:
                print(f"⚠️ Error analyzing {audio_file}: {e}")
    
    # Display results
    print("\nAvailable audio files and quality analysis:")
    print("-" * 80)
    
    good_files = []
    poor_files = []
    
    for idx, file in enumerate(wav_files, start=1):
        status = "Not transcribed"
        quality_info = ""
        
        if file in existing_analyses:
            analysis = existing_analyses[file]
            quality = analysis["quality"]
            status = f"Transcribed - Quality: {quality.upper()}"
            
            # Show speech parts info
            parts_info = []
            for part_name, metrics in analysis['parts'].items():
                if metrics['count'] > 0:
                    parts_info.append(f"{part_name.replace('_', ' ').title()}: {metrics['count']}")
            
            if quality in ["excellent", "good"]:
                good_files.append(file)
                paragraph_count = analysis['parts'].get('paragraphs', {}).get('count', 0)
                quality_info = f" ✅ Paragraph segments: {paragraph_count}"
                if parts_info:
                    quality_info += f" | Parts: {', '.join(parts_info)}"
            else:
                poor_files.append(file)
                if analysis['issues']:
                    quality_info = f" ⚠️ Issues: {', '.join(analysis['issues'][:2])}"
        
        print(f"{idx}. {file}")
        print(f"   Status: {status}{quality_info}")
        print()
    
    # Show recommendations
    if good_files:
        print("✅ RECOMMENDED FOR TRAINING:")
        for file in good_files:
            print(f"   • {file}")
        print()
    
    if poor_files:
        print("⚠️ NOT RECOMMENDED FOR TRAINING:")
        for file in poor_files:
            print(f"   • {file}")
        print()
    
    # User choice - support multiple selections
    print("📝 TRANSCRIPTION OPTIONS:")
    print("   • Enter single number (e.g., '3') to transcribe one file")
    print("   • Enter multiple numbers separated by commas (e.g., '1,2,5') for batch transcription")
    print("   • Enter range (e.g., '1-5') to transcribe files 1 through 5")
    print("   • Enter 'all' to transcribe all files")
    print("   • Enter 'q' to quit")
    
    choice = input("\nYour choice: ").strip()
    if choice.lower() == 'q':
        return
    
    # Parse user input to get selected file indices
    selected_indices = []
    
    try:
        if choice.lower() == 'all':
            selected_indices = list(range(len(wav_files)))
        elif '-' in choice and choice.count('-') == 1:
            # Range selection (e.g., "1-5")
            start, end = choice.split('-')
            start_idx = int(start.strip()) - 1
            end_idx = int(end.strip()) - 1
            if start_idx < 0 or end_idx >= len(wav_files) or start_idx > end_idx:
                print("❌ Invalid range.")
                return
            selected_indices = list(range(start_idx, end_idx + 1))
        elif ',' in choice:
            # Multiple selections (e.g., "1,2,5")
            numbers = [int(x.strip()) - 1 for x in choice.split(',')]
            for idx in numbers:
                if idx < 0 or idx >= len(wav_files):
                    print(f"❌ Invalid file number: {idx + 1}")
                    return
            selected_indices = numbers
        else:
            # Single selection
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(wav_files):
                print("❌ Invalid choice.")
                return
            selected_indices = [choice_idx]
    except ValueError:
        print("❌ Please enter valid numbers, ranges, or 'all'.")
        return

    selected_files = [wav_files[i] for i in selected_indices]
    
    print(f"\n🔊 Processing {len(selected_files)} file(s)...")
    for i, filename in enumerate(selected_files, 1):
        print(f"   {i}. {filename}")
    
    # Confirm if multiple files
    if len(selected_files) > 1:
        confirm = input(f"\nProceed with transcribing {len(selected_files)} files? (y/n): ").lower()
        if confirm != 'y':
            print("❌ Transcription cancelled.")
            return
    
    # Process each selected file
    batch_results = []
    
    for i, selected_file in enumerate(selected_files, 1):
        print(f"\n{'='*60}")
        print(f"🔊 Processing file {i}/{len(selected_files)}: {selected_file}")
        print(f"{'='*60}")
        
        try:
            json_path = transcribe_audio(selected_file)
            
            # Analyze quality of new transcription
            print("📊 Analyzing transcription quality and speech parts...")
            analysis = analyze_transcription_quality(json_path)
            
            # Store results for batch summary
            batch_results.append({
                'filename': selected_file,
                'analysis': analysis,
                'success': True
            })
            
            print(f"\n=== SPEECH PARTS ANALYSIS ===")
            for part_name, metrics in analysis['parts'].items():
                if metrics['count'] > 0:
                    print(f"{part_name.replace('_', ' ').title()}: {metrics['count']} segments "
                          f"(Conf: {metrics['avg_confidence']:.3f}, "
                          f"Dur: {metrics['avg_duration']:.1f}s, "
                          f"Words: {metrics['avg_words']:.1f})")
            
            print(f"\nQuality: {analysis['quality'].upper()} | "
                  f"Total Segments: {analysis['total_segments']}")
            
            if analysis['issues']:
                print(f"⚠️ Issues: {', '.join(analysis['issues'][:2])}")
            
            # Generate output files
            output_files = generate_outputs(json_path)
            
            print(f"✅ Completed processing {selected_file}")
            
        except Exception as e:
            print(f"❌ Error processing {selected_file}: {e}")
            batch_results.append({
                'filename': selected_file,
                'error': str(e),
                'success': False
            })
    
    # Batch summary
    print(f"\n{'='*60}")
    print(f"📋 BATCH TRANSCRIPTION SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in batch_results if r['success']]
    failed = [r for r in batch_results if not r['success']]
    
    print(f"Total files processed: {len(batch_results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n✅ SUCCESSFULLY TRANSCRIBED FILES:")
        
        training_suitable = []
        training_unsuitable = []
        
        for result in successful:
            analysis = result['analysis']
            filename = result['filename']
            quality = analysis['quality']
            paragraph_count = analysis['parts'].get('paragraphs', {}).get('count', 0)
            
            print(f"   • {filename}")
            print(f"     Quality: {quality.upper()} | Paragraphs: {paragraph_count} segments")
            
            if quality in ['excellent', 'good'] and paragraph_count > 0:
                training_suitable.append(filename)
            else:
                training_unsuitable.append(filename)
        
        # Training recommendations
        if training_suitable:
            print(f"\n🎯 RECOMMENDED FOR TRAINING ({len(training_suitable)} files):")
            for filename in training_suitable:
                print(f"   ✅ {filename}")
        
        if training_unsuitable:
            print(f"\n⚠️ NOT RECOMMENDED FOR TRAINING ({len(training_unsuitable)} files):")
            for filename in training_unsuitable:
                print(f"   ⚠️ {filename}")
    
    if failed:
        print(f"\n❌ FAILED TRANSCRIPTIONS:")
        for result in failed:
            print(f"   • {result['filename']}: {result['error']}")
    
    print(f"\n📚 TRAINING RECOMMENDATIONS:")
    print(f"   • Use paragraph segments from high-quality files for connected speech training")
    print(f"   • Use isolated word segments for word-level analysis")
    print(f"   • Use syllable segments for phoneme analysis")
    print(f"   • Compare repeated words with original paragraphs for consistency")

if __name__ == "__main__":
    main()
