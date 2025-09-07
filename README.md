# Thesis Transcript - Audio Transcription Tool

A Python tool for transcribing audio files using OpenAI's Whisper model, specifically designed for thesis research with Filipino/Tagalog audio content.

## Features

- 🎵 **Multi-format Audio Support**: WAV, MP3, M4A, FLAC, and more
- 🗣️ **Speech Part Detection**: Automatically identifies paragraphs, isolated words, syllables, and repeated words
- 📊 **Quality Analysis**: Confidence scores and utterance quality assessment
- 📝 **Detailed Output**: Timestamps, confidence scores, and structured transcriptions
- 🎯 **Filipino/Tagalog Support**: Optimized for Filipino language transcription
- 🔧 **Customizable Models**: Support for different Whisper model sizes

## Project Structure

```
thesisTranscript/
├── audio_input/           # Place your audio files here
├── transcript_output/     # Generated JSON and full transcription files
├── segments_output/       # Kaldi-style segment files
├── text_output/          # Kaldi-style text files
├── whisper-env/          # Python virtual environment
├── process_whisper.py    # Main transcription script
├── requirements.txt      # Python dependencies
├── pyproject.toml       # Modern project configuration
└── README.md           # This file
```

## Installation

### Prerequisites

1. **Python 3.8+**
2. **FFmpeg** (automatically installed via winget on Windows)
3. **Git** (optional, for cloning)

### Setup Instructions

1. **Clone or download the project**:
   ```bash
   git clone <your-repo-url>
   cd thesisTranscript
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Windows
   python -m venv whisper-env
   whisper-env\Scripts\activate

   # macOS/Linux
   python -m venv whisper-env
   source whisper-env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   # Using requirements.txt (recommended)
   pip install -r requirements.txt

   # Or using pyproject.toml
   pip install -e .

   # For GPU acceleration (if you have NVIDIA GPU)
   pip install -e .[gpu]

   # For development tools
   pip install -e .[dev]
   ```

## Usage

### Basic Usage

1. **Place audio files** in the `audio_input/` directory
2. **Run the transcription script**:
   ```bash
   python process_whisper.py
   ```
3. **Select the audio file** from the interactive menu
4. **Check results** in the output directories

### Advanced Usage

The script automatically:
- Detects speech parts (paragraphs, words, syllables)
- Analyzes utterance quality
- Generates multiple output formats
- Provides detailed statistics

### Output Files

For each transcribed audio file, you'll get:

- **JSON file**: Raw Whisper output with detailed segments
- **Full transcription**: Formatted text with timestamps and confidence scores
- **Segments file**: Kaldi-style segment mapping
- **Text file**: Kaldi-style utterance text
- **Speech parts analysis**: Automatic detection of different speech sections

## Speech Parts Detection

The tool automatically identifies four types of speech content:

1. **📖 Paragraphs/Sentences**: Connected speech (>6 words, >2 seconds)
2. **🔤 Isolated Words**: Single word utterances (1-3 words, <2 seconds)
3. **🔀 Syllables**: Very short utterances (<1 word equivalent)
4. **🔁 Repeated Words**: Words that appear in multiple parts

## Model Configuration

You can change the Whisper model for different accuracy/speed trade-offs:

```python
# In process_whisper.py, modify the MODEL variable:
MODEL = "base"     # Default - good balance
MODEL = "tiny"     # Fastest, basic accuracy
MODEL = "small"    # Better accuracy
MODEL = "medium"   # High accuracy (recommended for Filipino)
MODEL = "large"    # Best accuracy, slowest
```

## Quality Metrics

The tool provides quality analysis including:
- **Confidence scores**: Average log probability per utterance
- **Utterance length**: Word count and duration
- **Speech rate**: Words per minute
- **Quality assessment**: High/Medium/Low quality classification

## Language Support

While designed for Filipino/Tagalog, the tool supports:
- Automatic language detection
- Manual language specification
- Multilingual content

To specify language manually, modify the script:
```python
# For Filipino/Tagalog
--language tl

# For English
--language en

# For automatic detection
--language None
```

## System Requirements

### Minimum
- Python 3.8+
- 4GB RAM
- 2GB free disk space

### Recommended
- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU with CUDA support (for faster processing)
- 10GB+ free disk space

## Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   ```bash
   # Windows (install via winget)
   winget install Gyan.FFmpeg
   
   # macOS (install via Homebrew)
   brew install ffmpeg
   
   # Linux (install via package manager)
   sudo apt install ffmpeg  # Ubuntu/Debian
   ```

2. **CUDA/GPU issues**:
   - Install CUDA-compatible PyTorch versions
   - Use CPU-only versions if no GPU available

3. **Memory errors**:
   - Use smaller Whisper models (`tiny`, `base`)
   - Process shorter audio files
   - Close other applications

4. **Audio format issues**:
   - Convert to WAV format first
   - Check audio file integrity
   - Ensure proper sample rate (16kHz recommended)

## Development

### Running Tests
```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Run with coverage
pytest --cov=process_whisper
```

### Code Formatting
```bash
# Format code
black process_whisper.py

# Sort imports
isort process_whisper.py

# Lint code
flake8 process_whisper.py
```

**Happy transcribing! 🎙️✨**
