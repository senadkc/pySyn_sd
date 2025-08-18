# py_syn_sd


# Resource-Efficient Python Syntax Error Repair (BiLSTM + LSTM)

_A lightweight Automatic Program Repair (APR) pipeline for Python syntax errors with hybrid fault localization, BiLSTM error type classification, and LSTM token-level fixing._

![Workflow](docs/figures/flowchart.png)

**TL;DR**
- **Dataset:** 6k synthetic faulty samples  
- **Error Type Classification (BiLSTM):** Accuracy ≈ 98.0% (±2.8), F1 ≈ 93.7% (±2.8)  
- **Repair (LSTM):** Accuracy ≈ 98.4% (±0.3), F1 ≈ 98.9% (±0.3)  
- **End-to-end repair (Test-1, n=925):** 85.84% success  
- **Efficiency:** ~32× faster inference than CodeT5, ~1540× smaller model size  

---

## 1) Repository Structure

```

.
├─ full-test.py                         # End-to-end APR demo
├─ lstm\_metrics\_std\_runs.py             # LSTM repair metrics across runs
├─ text\_classification\_metrics\_runs.py  # BiLSTM classification metrics                     
├─ docs/
│  └─ figures/                          # Exported PNG figures & tables from PDF
└─ README.md

````

---

## 2) Environment

```bash
conda create -n apr-python38 python=3.8
conda activate apr-python38

pip install numpy==1.23.5 scikit-learn==1.2.2 tensorflow==2.12.0 matplotlib==3.7.2
pip install astunparse astroid gast

# (Optional) for semantic localization:
npm install -g pyright
````

**System (example):** Python 3.8, TensorFlow 2.12.0, Intel i9-10900X, 2×RTX 3090 (24GB)
**Training time (example):** \~1h24m (BiLSTM), \~1h27m (LSTM)

---

## 3) Data

* **Training:** 2,000 correct + 6,000 faulty (2k Missing, 2k Extra, 2k Incorrect)
* **Test-1 (Synthetic):** 925 samples (300/300/300 + 25 indentation)
* **Test-2 (CodeNet-derived):** 600 faulty variants (200×3)
* Synthetic generation is mutation-based (see Algorithm 1 in paper)

---

## 4) Usage

### 4.1 Error Type Classification (BiLSTM)

```bash
python text_classification_metrics_runs.py \
  --data_dir data \
  --batch_size 256 --epochs 150 --seed 42
```

![BiLSTM Architecture](docs/figures/fig3_bilstm_arch.png)
![Table 4: BiLSTM Metrics](docs/figures/table4_bilstm_metrics.png)

### 4.2 Token Repair (LSTM)

```bash
python lstm_metrics_std_runs.py \
  --data_dir data \
  --batch_size 256 --epochs 250 --seed 42
```

![LSTM Architecture](docs/figures/fig4_lstm_arch.png)
![Table 7: LSTM Metrics](docs/figures/table7_lstm_metrics.png)

### 4.3 End-to-End Demo

```bash
python full-test.py --input path/to/faulty.py --output repaired.py
```

*Pipeline:* Tokenization → Error type (BiLSTM) → Fault localization (AST; fallback Pyright) → Token repair (LSTM) → Validation

---

## 5) Evaluation

### 5.1 Synthetic Test-1 (n=925)

| Error Type | Total | Fixed | Success |
| ---------- | ----: | ----: | ------: |
| Missing    |   300 |   258 |   86.0% |
| Extra      |   300 |   262 |   87.3% |
| Incorrect  |   300 |   252 |   84.0% |

* Parentheses/quotes mismatches: **274/274 = 100%** fixed
* Indentation errors (n=25): **22/25 = 88%** (rule-based baseline ≈ 60%)

### 5.2 CodeNet-derived Test-2 (n=600)

* “Unexpected Indent”: **90%**
* “Unindent mismatch”: **100%**
* “Unexpected EOF”: **75%**
* “Invalid syntax (overall)”: **59.8%**

![Detailed Results by Type](docs/figures/table11_results.png)

---

## 6) Efficiency vs. Transformers

| Model    | Inference Time (ms) | Params (M) | Repair Success |
| -------- | ------------------: | ---------: | -------------: |
| **Ours** |               8,651 |        0.5 |            57% |
| PLBart   |              10,701 |        140 |            23% |
| CodeGen  |              81,536 |        350 |            53% |
| CodeT5   |             275,041 |        770 |            27% |

![Efficiency & Comparison](docs/figures/fig8_efficiency.png)

---

## 7) Method Overview

* **Hybrid Fault Localization:** AST-based structural error detection; fallback to Pyright for semantic checks → reduced search space
* **BiLSTM Error Classification:** Three classes (Missing / Extra / Incorrect); embeddings + bidirectional context + softmax
* **LSTM Token Repair:** Embedding → stacked LSTM layers → token repair suggestions → iterative retries until valid fix

![Error Examples](docs/figures/fig5_7_error_examples.png)


## 8) Acknowledgements

We thank Bursa Technical University High-Performance Computing Laboratory for support.

```
