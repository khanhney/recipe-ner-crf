# Recipe NER: Identifying Key Entities in Recipe Data

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CRF](https://img.shields.io/badge/Model-CRF-green.svg)](https://sklearn-crfsuite.readthedocs.io/)
[![spaCy](https://img.shields.io/badge/spaCy-3.5+-orange.svg)](https://spacy.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

> A Custom Named Entity Recognition (NER) model using Conditional Random Fields (CRF) to extract **ingredients**, **quantities**, and **units** from recipe text.

---

## 🎯 Project Overview

This project develops an NER system that automatically identifies and extracts key entities from recipe ingredient lists, enabling structured data extraction from unstructured recipe text.

### Entity Types

| Entity | Description | Examples |
|--------|-------------|----------|
| 🥕 **Ingredient** | Food items, spices | rice, onion, turmeric, chicken |
| 🔢 **Quantity** | Numeric amounts | 2, 1/2, 3-1/2, 500 |
| 📏 **Unit** | Measurement units | cups, tablespoon, grams, pinch |

### Example

```
Input:  "2 cups rice 1 tablespoon oil 3 cloves garlic"

Output:
  2          → quantity
  cups       → unit
  rice       → ingredient
  1          → quantity
  tablespoon → unit
  oil        → ingredient
  3          → quantity
  cloves     → unit
  garlic     → ingredient
```

---

## 💼 Business Value

| Application | Benefit |
|-------------|---------|
| **Recipe Search Engines** | Find recipes by specific ingredients |
| **Meal Planning Apps** | Auto-generate shopping lists |
| **Dietary Tracking** | Extract nutritional components |
| **E-commerce Integration** | Auto-populate grocery carts |
| **Voice Assistants** | Enable hands-free cooking guidance |

---

## 📁 Repository Contents

```
recipe-ner-crf/
├── recipe-ner-crf.ipynb          # Main Jupyter notebook with full implementation
├── recipe-ner-crf-report.pdf     # Detailed project report
└── README.md                      # This file
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/recipe-ner-crf.git
cd recipe-ner-crf

# Install dependencies
pip install sklearn-crfsuite==0.5.0
pip install spacy pandas numpy matplotlib seaborn scikit-learn joblib

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Dependencies

```
sklearn-crfsuite==0.5.0
spacy>=3.5.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
joblib>=1.0.0
```

---

## 🚀 Quick Start

### Run the Notebook

```bash
jupyter notebook recipe-ner-crf.ipynb
```

### Use the Trained Model

```python
import joblib

# Load trained model
crf = joblib.load('crf_model.pkl')

# Parse recipe
recipe = "2 cups flour 1 tablespoon sugar 3 eggs"
tokens = recipe.split()
features = sent2features(tokens)  # Feature extraction function from notebook

# Predict
labels = crf.predict([features])[0]

# Display results
for token, label in zip(tokens, labels):
    print(f"{token}: {label}")
```

---

## 📊 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CRF MODEL PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: "2 cups rice 1 tablespoon oil"                      │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │  Tokenization   │  → ['2', 'cups', 'rice', ...]         │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │    Feature      │  → 25+ features per token              │
│  │   Extraction    │     • Core (token, lemma, POS)         │
│  │                 │     • Domain (is_unit, is_quantity)    │
│  │                 │     • Context (prev/next token)        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐                                        │
│  │   CRF Model     │  → Sequence labeling with              │
│  │   Prediction    │     learned transitions                │
│  └────────┬────────┘                                        │
│           │                                                  │
│           ▼                                                  │
│  Output: [quantity, unit, ingredient, ...]                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Feature Engineering

### Features Used (25+)

| Category | Features | Purpose |
|----------|----------|---------|
| **Core** | token, lemma, pos_tag, shape, is_digit, is_title, is_upper | Lexical properties |
| **Domain** | is_quantity, is_unit, is_fraction, is_decimal, is_numeric | Recipe-specific patterns |
| **Context** | prev_token, next_token, prev_is_quantity, next_is_unit, BOS, EOS | Surrounding context |

### Keyword Sets

```python
# Unit keywords
unit_keywords = {'cup', 'cups', 'tablespoon', 'teaspoon', 'gram', 'grams',
                 'kg', 'ml', 'pinch', 'sprig', 'clove', 'inch', ...}

# Quantity pattern (regex)
quantity_pattern = r'^(\d+[-]?\d*\/?\d*|\d+\.\d+|\d+)$'
# Matches: "2", "1/2", "2-1/2", "3.5", "500"
```

---

## 📈 Model Performance

### Overall Metrics

| Dataset | Accuracy | F1 (Weighted) | F1 (Macro) |
|---------|----------|---------------|------------|
| Training | ~95% | ~0.95 | ~0.93 |
| Validation | ~90% | ~0.90 | ~0.88 |

### Per-Class Performance

| Label | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Ingredient | 0.92 | 0.95 | 0.93 |
| Quantity | 0.89 | 0.85 | 0.87 |
| Unit | 0.88 | 0.82 | 0.85 |

---

## 📚 Methodology

### 1. Data Preparation
- Load JSON dataset with recipe ingredients and labels
- Tokenize input text and validate label alignment
- Clean data by removing mismatched records

### 2. Exploratory Data Analysis
- Analyze token distribution across labels
- Identify top ingredients and units
- Detect class imbalance (ingredients ~70%)

### 3. Feature Extraction
- Define unit/quantity keyword sets
- Implement `word2features()` with 25+ features
- Apply class weighting for imbalanced data

### 4. Model Training
- Initialize CRF with L-BFGS optimization
- Apply L1 (c1=0.5) and L2 (c2=1.0) regularization
- Train on 70% data, validate on 30%

### 5. Evaluation & Error Analysis
- Generate classification reports
- Create confusion matrices
- Analyze misclassification patterns

---

## 🔍 Key Findings

### Insights from EDA
1. **Top Ingredients**: Salt, onion, garlic, cumin, turmeric (Indian cuisine focus)
2. **Top Units**: teaspoon, tablespoon, cup, grams
3. **Class Imbalance**: Ingredients dominate (~70%) requiring weighted training

### Common Error Patterns
- Ambiguous tokens (e.g., "clove" can be unit or ingredient)
- Word-form quantities (e.g., "one", "half")
- Boundary errors at sequence start/end

---

## 🚧 Limitations

| Limitation | Impact |
|------------|--------|
| English only | Cannot process other languages |
| Indian cuisine focus | May not generalize to all cuisines |
| No BIO tagging | Cannot identify multi-word entities |
| Linear model | Limited feature interactions |

---

## 🔮 Future Improvements

- [ ] Implement BiLSTM-CRF for better sequence modeling
- [ ] Add word embeddings (Word2Vec, GloVe, BERT)
- [ ] Support multi-word entities with BIO tagging
- [ ] Multilingual support for global recipes
- [ ] Deploy as REST API
- [ ] Build interactive web demo

---

## 📖 Report

For detailed analysis, visualizations, and insights, see the full report:
- 📄 [recipe-ner-crf-report.pdf](recipe-ner-crf-report.pdf)

---

## 🛡️ License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **sklearn-crfsuite** - CRF implementation
- **spaCy** - NLP feature extraction

---

## 📬 Contact

For questions or feedback, please open an issue in this repository.

---

<p align="center">
  <b>⭐ Star this repo if you find it useful! ⭐</b>
</p>
