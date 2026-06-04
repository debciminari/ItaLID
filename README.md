# No One-Size-Fits-All Solution: Evaluating Language Identification Methods for Languages of Italy

## 1. Overview

We present a study on Language Identification (LID) focused on Italian (*it*) and eight closely related language varieties spoken in Italy: Corsican (*co*), Emiliano-Romagnol (*eml*), Lombard (*lmo*), Neapolitan (*nap*), Piedmontese (*pms*), Sardinian (*sc*), Sicilian (*scn*), and Venetian (*vec*).  

The task is challenging due to strong linguistic similarity among languages and the low-resource nature of most languages. To address these challenges, we investigate a range of modeling approaches and feature representations, including word-level features and character n-grams. We construct a comparable corpus from Wikipedia and evaluate Multinomial Naïve Bayes, fastText, and Minerva LLM. We further assess robustness in an out-of-domain scenario using poetry data.

---

## 2. Models

We evaluate three families of models:

- **Multinomial Naïve Bayes (MNB)**
  - Word-level n-grams;
  - Character n-grams (1–6);
  - Feature ablation and combination studies.

- **fastText**
  - Efficient embedding-based classifier;
  - Strong baseline for multilingual and low-resource settings.

- **Minerva-7B-Instruct**
  - Different prompting strategies:
    - *easy*: direct classification;
    - *medium*: constrained label set;
    - *hard*: structured and more restrictive prompting.

---

## 3. Data

The dataset is constructed from Wikipedia and includes:

- `train.json`: training set
- `test.json`: evaluation set
- `wiki_qids.csv`: Wikipedia QID mapping and metadata
- `query.csv`: query used to retrieve intersection articles across  languages

---

## 4. Out-of-Domain Evaluation

We evaluate model robustness on out-of-domain data, specifically poetry, to assess generalisation beyond Wikipedia-style text.

---

## 5. Repository Structure

- **data/**
  - `train.json`, `test.json`, `wiki_qids.csv`, `query.csv`

- **mnb/**
  - Scripts for Multinomial Naïve Bayes experiments with multiple feature configurations.

- **fast/**
  - fastText training and evaluation scripts

- **LLM/**
  - Prompting experiments:
    - `easy`: direct prompting
    - `medium`: constrained label space
    - `hard`: structured prompting

- **out-of-domain/**
  - Script for evaluation on poetry (out-of-domain setting)

---

## 6. References

- Joulin et al. (2017) — fastText
- - Orlando et al. (2024) — Minerva-7B-Instruct

---

## 7. License

Add license information if applicable.
