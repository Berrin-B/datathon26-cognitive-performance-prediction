
# Datathon'26: Cognitive Performance Prediction

This repository contains the end-to-end Machine Learning pipeline developed for the **Yapay Zeka ve Teknoloji Akademisi Datathon26 (Artificial Intelligence and Technology Academy Datathon26)**. The core objective of this project is to analyze human cognitive metrics and predict cognitive performance scores by leveraging advanced data preprocessing, feature engineering, and ensemble modeling techniques.

## 🚀 Project Architecture

The project is structured into three main phases executed through specialized Jupyter Notebooks:

1. **`01-data-cleaning.ipynb` (Data Preprocessing & Wrangling)**
   * Handled missing values, treated outliers, and engineered domain-specific features.
   * Scaled, encoded, and transformed raw behavioral/cognitive data into a model-ready format.

2. **`EDA (1).ipynb` (Exploratory Data Analysis)**
   * Conducted deep statistical data analysis to uncover underlying patterns, correlations, and anomalies.
   * Utilized advanced visualization libraries to analyze feature distributions and investigate how different cognitive parameters correlate with performance metrics.

3. **`final-model.ipynb` (Model Development & Evaluation)**
   * Built, tuned, and evaluated predictive machine learning models.
   * Applied feature selection, hyperparameter optimization, and ensemble techniques to maximize prediction accuracy and robustness.

## 📈 Executive Summary & Performance Breakdown

The predictive pipeline evaluates multiple advanced gradient boosting frameworks and combines their generalization capabilities through a meta-regressor. The final performance matrix stands as follows:

| Model Architecture | Validation Score (OOF RMSE) | Key Mechanics |
| :--- | :---: | :--- |
| **Stacking Ensemble (Meta-Model)** | 🏆 **1.2174** | **Ridge Regression Meta-Learner (Weights: CatBoost: ~0.78, XGBoost: ~0.22)** |
| CatBoost Seed Averaging | 1.2177 | 10-Seed Average combined with Optuna-tuned hyperparameters |
| XGBoost Seed Averaging | 1.2219 | 10-Seed Average utilizing historical tree-building methods |

---

## 🚀 Machine Learning Pipeline Architecture

The workflow is meticulously structured into five core modular phases:

### 1. Advanced Data Cleaning & Robust Preprocessing
* **Missing Value Imputation:** Integrated multi-level strategy handling systemic missingness (e.g., categorical fields filled with statistical modes or domain-specific string indicators; numerical missingness resolved via robust conditional transformations, such as computing `stres_skoru` medians grouped by `ruh_sagligi_durumu`).
* **Outlier Mitigation:** Executed feature clipping using extreme quantiles (1st and 99th percentiles) on volatile physical and physiological parameters (`vucut_kitle_indeksi`, `gunluk_adim_sayisi`, `uykuya_dalma_suresi_dk`, etc.) to prevent variance spikes during tree node splits.

### 2. High-Dimensional Feature Engineering (Expanding to 82+ Features)
A comprehensive feature creation strategy was deployed to capture non-linear, multi-variable interactions within human behavior:
* **Sleep Quality Indexes:** Modeled sleep architecture consistency via features like `toplam_onarici_uyku` (REM + Deep Sleep ratios), `uyku_surekliligi_bozuklugu`, and sleep depth balance ratios.
* **Biometric & Stress Mapping:** Synthesized cognitive load matrices including `zihinsel_yuk` (working hours $\times$ stress), `stres_aktivite_orani`, and `stres_nabiz`.
* **Chronotype/Circadian Alignment:** Designed synchronization flags mapping chronotype alignment (`kronotip_senkron`) with regular weekday/weekend occupational dynamics.
* **Environmental/Physical Impact Variables:** Formulated a localized clinical score proxy (`psqi_benzeri_skor`) capturing structural sleep metrics, screen time exposure, caffeine latency, and thermal deviation factors.

### 3. Cross-Validated Target Encoding
* Applied out-of-fold categorical target encoding utilizing a **5-fold Stratified K-Fold** setup across complex high-cardinality categorical strings and interaction combinations (e.g., `mevsim_kronotip`, `ruh_stres_combo`). This approach effectively minimized target leakage while maximizing statistical representativeness.

### 4. Automated Hyperparameter Tuning via Optuna
* Executed sequential hyperparameter tuning using **Optuna** over 30 targeted exploration trials targeting the `CatBoostRegressor`. The optimized configuration targeted deep architectural parameters:
  ```python
  Best_Params = {
      'iterations': 4834, 'learning_rate': 0.02658, 'depth': 5, 
      'l2_leaf_reg': 6.978, 'bagging_temperature': 0.427, 
      'random_strength': 1.517, 'border_count': 101
  }
## 📬 Contact 

Developed as part of the **Yapay Zeka ve Teknoloji Akademisi Data Science & AI Fellowship**. Feel free to open an issue or reach out for any questions regarding the dataset or the implementation details!
