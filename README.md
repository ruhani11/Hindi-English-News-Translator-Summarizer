Hindi to English News Summarizer
This project is a domain-specific NLP pipeline that performs translation and summarization of Hindi news content into English. It combines web scraping, neural machine translation (NMT), and abstractive summarization into a streamlined solution for cross-lingual news accessibility.

🔍 Features
Scrapes Hindi news headlines and articles

Translates Hindi content to English using a fine-tuned MarianMT model (Helsinki-NLP/opus-mt-hi-en)

Generates concise English summaries using a BART-based summarization model

Simple Flask-based web app for user interaction

📊 Model Stack
MarianMT (Hindi-English translation)

Facebook BART (Summarization)

Hugging Face Transformers & Datasets

Training and evaluation using BLEU and ROUGE metrics

🛠 Tools & Frameworks
Python, Flask, BeautifulSoup, Pandas

Hugging Face Transformers

Jupyter Notebooks for experimentation and visualization
