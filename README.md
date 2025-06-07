# 📰 Hindi to English News Summarizer

This project is a domain-specific NLP pipeline that performs translation and summarization of Hindi news content into English. It combines web scraping, neural machine translation (NMT), and abstractive summarization into a streamlined solution for cross-lingual news accessibility.

---

## 🔍 Features

- Scrapes Hindi news headlines and articles
- Translates Hindi content to English using a fine-tuned MarianMT model (`Helsinki-NLP/opus-mt-hi-en`)
- Generates concise English summaries using a BART-based summarization model
- Provides a simple Flask-based web app for user interaction

---

## 🧠 Model Stack

- **MarianMT**: Hindi-English translation (`Helsinki-NLP/opus-mt-hi-en`)
- **Facebook BART**: English summarization

---

## 📦 Frameworks and Libraries

- Hugging Face Transformers & Datasets
- Python
- Flask
- BeautifulSoup
- Pandas

---

## 📊 Evaluation Metrics

- BLEU
- ROUGE

---

## 🧪 Development Tools

- Jupyter Notebooks for experimentation and visualization

---

## 🚀 Quick Overview

1. **Web Scraping**: Collects Hindi news data
2. **Translation**: Uses MarianMT to convert Hindi to English
3. **Summarization**: Applies BART model to generate summaries
4. **Web Interface**: Allows users to input and see outputs via Flask app

---

## 🔗 Future Improvements

- Add multilingual support
- Incorporate more domains (e.g., sports, finance)
- Add user feedback loop for better fine-tuning

