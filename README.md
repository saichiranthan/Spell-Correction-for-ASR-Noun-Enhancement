# Spell Correction for ASR Noun Enhancement

A comprehensive NLP project focused on correcting Automatic Speech Recognition (ASR) errors, particularly for medication names and medical terminology in healthcare conversations.

## Project Overview

This project implements both traditional baseline models and advanced transformer-based approaches to correct spelling errors in ASR-generated medical transcriptions. The system specifically targets noun correction, with emphasis on medical terminology and drug names.

## Problem Statement

ASR systems often struggle with:
- Proper nouns and medication names
- Medical terminology
- Domain-specific vocabulary
- Segmentation errors (e.g., "healthcare" → "health care")
- Phonetic substitutions

## Dataset

- **Source**: Pre-prepared ASR transcription pairs from medical conversations
- **Size**: 10,000 sentence pairs
- **Split**: 70% training, 15% validation, 15% test
- **Focus**: Medical terminology, medication names, and general medical nouns

## Models Implemented

### Baseline Models
1. **Levenshtein Distance Model**
   - Dictionary-based correction using edit distance
   - Maximum edit distance threshold: 2
   - Simple word-by-word correction approach

2. **N-gram Language Model**
   - Bigram context-aware correction
   - Combines edit distance with contextual probability
   - Uses frequency statistics from training corpus

### Advanced Models
1. **T5-base (Text-to-Text Transfer Transformer)**
   - General-purpose sequence-to-sequence model
   - Fine-tuned on medical transcription correction task
   - 15 epochs training with early stopping

2. **SciFive-base-Pubmed_PMC**
   - Domain-specific T5 model pre-trained on biomedical texts
   - Specialized medical vocabulary knowledge
   - Superior performance on medical terminology

## Key Features

- **Comprehensive Error Analysis**: Identifies and categorizes ASR error patterns
- **Multi-metric Evaluation**: Sentence accuracy, word accuracy, noun accuracy, CER, BLEU-4
- **Medical Focus**: Specialized handling of medication names and medical terms
- **Context-Aware**: Advanced models consider full sentence context
- **Scalable Architecture**: Modular design for easy extension

## Results Summary

| Model | Sentence Accuracy (%) | Word Accuracy (%) | Noun Accuracy (%) | CER (%) |
|-------|----------------------|-------------------|-------------------|---------|
| Levenshtein (Baseline) | 7.00 | 68.64 | 73.93 | 4.00 |
| N-gram (Baseline) | 6.53 | 68.53 | 73.52 | 4.19 |
| T5-base (Advanced) | 8.27 | 83.28 | 75.91 | 2.89 |
| **SciFive-base (Advanced)** | **12.33** | **83.78** | **76.97** | **3.01** |

## Project Structure

```
├── dataset/
│   ├── Spell_Correction_for_ASR_Noun_Enhancement_assignment_dataset.xlsx
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── models/
│   ├── t5_base_results/ (not uploaded due to size constraints)
│   └── SciFive-base-results/ (not uploaded due to size constraints)
├── notebooks/
│   ├── 01_Data_Exploration_and_Preprocessing.ipynb
│   ├── 02_Baseline_Model.ipynb
│   ├── 03_Advanced_Model_T5.ipynb
│   └── 04_Final_Evaluation_and_Analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── baseline_models.py
│   ├── advanced_models.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

## Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/saichiranthan/spell-correction-asr-noun-enhancement.git
cd spell-correction-asr-noun-enhancement
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Running the Complete Pipeline

1. **Data Preprocessing**
```python
from src.data_preprocessing import preprocess_data
train_df, val_df, test_df = preprocess_data('dataset/raw_data.xlsx')
```

2. **Baseline Models**
```python
from src.baseline_models import LevenshteinCorrector, NgramCorrector

# Levenshtein correction
corrector = LevenshteinCorrector(vocabulary)
corrected_text = corrector.correct_sentence("misspeled text")

# N-gram correction
ngram_corrector = NgramCorrector(vocabulary, bigram_counts)
corrected_text = ngram_corrector.correct_sentence("misspeled text")
```

3. **Advanced Models**
```python
from src.advanced_models import T5Corrector

# Load fine-tuned model
corrector = T5Corrector('models/SciFive-base-results/best_model')
corrected_text = corrector.correct("misspeled medical text")
```

### Training Custom Models

```python
# Train T5 model
from src.advanced_models import train_t5_model

trainer, model_path = train_t5_model(
    model_checkpoint="t5-base",
    train_dataset=train_ds,
    val_dataset=val_ds,
    output_dir="models/custom_t5"
)
```

## Evaluation Metrics

- **Sentence Accuracy**: Percentage of perfectly corrected sentences
- **Word Accuracy**: Percentage of correctly predicted words
- **Noun Accuracy**: Percentage of medical nouns correctly identified
- **Character Error Rate (CER)**: Edit distance normalized by sentence length
- **BLEU-4 Score**: Sequence similarity measure

## Key Findings

1. **SciFive-base outperforms all other models** due to specialized medical vocabulary
2. **Segmentation errors are the primary challenge** (e.g., "healthcare" → "health care")
3. **Context-aware models significantly outperform word-level corrections**
4. **Medical noun accuracy is higher than general word accuracy** across all models
5. **Transformer models show consistent performance across sentence lengths**

## Technical Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for transformer training)
- 16GB+ RAM
- Google Colab compatible

## Future Improvements

- [ ] Apply data augmentation (synthetic noise injection, phonetic similarity errors) to improve robustness.
- [ ] Explore larger transformer models (e.g., T5-large, BioT5, GPT-style models) to test scalability and performance gains.
- [ ] Implement confidence scoring for corrections
- [ ] Experiment with reinforcement learning or constrained decoding to prioritize correct noun predictions

## Contributors

Sai Chiranthan H M

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Assignment provided by [Augnito AI]
- SciFive model by [razent/SciFive-base-Pubmed_PMC]
- Hugging Face Transformers library
- spaCy for NLP processing
