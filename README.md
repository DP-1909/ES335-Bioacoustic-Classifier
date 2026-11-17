# Automated Wildlife & Environmental Sound Classification (ESC-50)

**Course:** ES 335 — Machine Learning (Indian Institute of Technology Gandhinagar)  
**Authors:** Bikram Maurya (24110077) & Dev Kaushik Panchal (24110101)  
**Live Interactive Website:** [https://dp-1909.github.io/ES335-Bioacoustic-Classifier/](https://dp-1909.github.io/ES335-Bioacoustic-Classifier/)  
**Google Colab Notebook:** [View Training & Evaluation Notebook](https://colab.research.google.com/drive/14B-Nmj8uBAKvaXA5gN1hlQxYnRp_NM6H?usp=sharing)

---

## 1. Project Overview & Research Question

Passive acoustic monitoring enables non-invasive biodiversity tracking across remote habitats without human field observers, but requires compact models that distinguish overlapping biological vocalizations from ambient weather textures.

* **Research Question:** Can compact, time-averaged spectral and cepstral descriptors ($17$ handcrafted acoustic features) reliably classify $19$ wildlife and natural soundscape categories without deep neural networks, and how strongly does decision-boundary geometry (linear vs. kernel vs. tree ensemble) govern multi-class separability?
* **Unit of Analysis ($N = 760$):** Each row represents one $5\text{-second}$ mono WAV recording ($44.1\text{ kHz}$), compressed into a $17\text{-dimensional}$ continuous feature vector $\mathbf{x}_i \in \mathbb{R}^{17}$ paired with a discrete target label $y_i \in \{1, \dots, 19\}$.
* **Prediction-Time Setup:** An edge acoustic sensor buffers a $5\text{s}$ audio window, extracts the $17$ descriptors on-device, and outputs class probabilities—requiring zero metadata or future labels.

---

## 2. Dataset & Feature Extraction

* **Source:** Curated from Karol Piczak's public CC-licensed [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50) (2,000 recordings across 50 environmental classes).
* **Balanced Subset ($N = 760$):** All $19$ wildlife and natural environment classes were extracted ($40\text{ clips/class}$; random-guess baseline $= 1/19 = 5.26\%$), preserving the $5$ official ESC-50 predefined folds ($8\text{ clips/class/fold} = 152\text{ clips/fold}$) so takes from the same source recording never leak across train and test splits.
* **Target Categories (19):**
  * *Wildlife & Animals (10):* `dog`, `rooster`, `pig`, `cow`, `frog`, `cat`, `hen`, `insects`, `sheep`, `crow`
  * *Birds, Insects & Nature (9):* `chirping_birds`, `crickets`, `rain`, `sea_waves`, `crackling_fire`, `water_drops`, `wind`, `pouring_water`, `thunderstorm`
* **17 Extracted Acoustic Features (`librosa`):**
  1. **MFCC 1–13 (13 features):** Mean Mel-Frequency Cepstral Coefficients capturing vocal-tract resonance and timbral envelope.
  2. **Spectral Centroid (1 feature):** Spectral center-of-mass in Hz (separating high-frequency stridulation from low-frequency rumbles).
  3. **Spectral Bandwidth (1 feature):** Frequency spread around the centroid in Hz (distinguishing narrowband harmonic calls from broadband noise).
  4. **Zero-Crossing Rate (1 feature):** Waveform sign-change rate (capturing unvoiced sibilants, insect stridulation, and percussive crackles).
  5. **RMS Energy (1 feature):** Root-mean-square acoustic intensity (distinguishing sparse transients from sustained textures).

---

## 3. Evaluation Protocol & Benchmark Results

All models were evaluated using **leak-free 5-fold nested cross-validation**: each outer iteration holds out $1$ complete ESC-50 predefined fold ($152\text{ clips}$) for testing while an inner $4\text{-fold}$ `GridSearchCV` optimizes **Macro-F1** on the remaining $4\text{ folds}$ ($608\text{ clips}$). `StandardScaler` is fitted strictly inside each training fold via `sklearn.pipeline.Pipeline`.

| Model Family | Preprocessing | Tuned Hyperparameters | Overall Accuracy | Overall Macro-F1 |
| :--- | :--- | :--- | :---: | :---: |
| **SVM (RBF Kernel) — Champion** | `StandardScaler` (in-fold) | `C = 1`, `gamma = 'scale'` | **56.84%** | **56.68%** |
| **Random Forest** | Raw / Scale-Invariant | `n_estimators = 200`, `min_samples_split = 5` | 51.45% | 50.74% |
| **Logistic Regression (Baseline)** | `StandardScaler` (in-fold) | `penalty = 'l2'`, `C = 1` | 50.92% | 50.69% |

* **Key Finding:** Mapping standardized acoustic descriptors into a non-linear RBF kernel space improves Macro-F1 by **+5.99 percentage points** over the linear Logistic Regression baseline ($56.68\%$ vs. $50.69\%$).
* **Primary Error Modes:** Stationary broadband water/wind textures (`rain`, `pouring_water`, `sea_waves`, `wind`) and high-frequency insect/bird stridulations (`crickets`, `insects`, `chirping_birds`) account for the majority of off-diagonal confusions due to clip-level mean pooling discarding temporal onset dynamics.

---

## 4. Repository Structure

```text
ES335-Bioacoustic-Classifier/
├── index.html          # Semantic academic report & interactive dashboard layout
├── styles.css          # Editorial typography & responsive warm-ivory/forest styling
├── app.js              # Client-side JS inference engine (LR, RBF-SVM, RF), PCA plot & confusion matrices
├── site_data.json      # Nested CV metrics, 19x19 confusion matrices, PCA projections & clip metadata
├── models_export.json  # Exported scaler parameters, LR weights, RBF support vectors & 200 RF trees
├── audio/              # 19 representative 5-second ESC-50 WAV recordings (1 per class)
└── README.md           # Project documentation & reproduction instructions
```

---

## 5. Reproducibility & Local Execution

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/DP-1909/ES335-Bioacoustic-Classifier.git
   cd ES335-Bioacoustic-Classifier
   ```
2. **Run the Training & Export Pipeline:**
   * Open the [Google Colab Notebook](https://colab.research.google.com/drive/14B-Nmj8uBAKvaXA5gN1hlQxYnRp_NM6H?usp=sharing) (or run locally with Python 3.10+, `librosa`, `scikit-learn`, `numpy`, and `pandas`).
   * Download the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50), filter the $19$ target categories ($760\text{ clips}$), extract the $17$ features, run 5-fold nested `GridSearchCV`, and export `site_data.json` and `models_export.json`.
3. **Launch the Interactive Web Application Locally:**
   ```bash
   python -m http.server 8000
   ```
   Open `http://localhost:8000` in any modern web browser to audition the 19 recordings and test live feature perturbations across all three classifiers.

---

## 6. Team Contributions, Disclosure & Citations

* **Bikram Maurya (`24110077` — `bikram.maurya@iitgn.ac.in`):** ESC-50 dataset curation and quality auditing, 2D PCA exploratory analysis, Logistic Regression baseline benchmarking, and confusion-matrix error analysis.
* **Dev Kaushik Panchal (`24110101` — `dev.panchal@iitgn.ac.in`):** `librosa` 17-feature extraction pipeline, leak-free 5-fold nested cross-validation setup, RBF-SVM/Random Forest hyperparameter tuning, and client-side JSON inference serialization.
* **AI & Tools Disclosure:** Built with Python (`librosa`, `scikit-learn`, `numpy`, `pandas`), KaTeX, and vanilla HTML5/CSS3/ES6 JavaScript. Generative AI assistance (Google Gemini) was used to help structure the static web interface and debug JSON model serialization; all experimental splits, metrics, and model parameters were independently executed and verified by the authors.
* **References:**
  1. K. J. Piczak, *"ESC: Dataset for Environmental Sound Classification,"* Proc. ACM Multimedia, pp. 1015–1018, 2015.
  2. B. McFee et al., *"librosa: Audio and Music Signal Analysis in Python,"* Proc. SciPy, 2015.
  3. F. Pedregosa et al., *"Scikit-learn: Machine Learning in Python,"* JMLR 12, pp. 2825–2830, 2011.