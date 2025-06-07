import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import MarianMTModel, MarianTokenizer, BartTokenizer, BartForConditionalGeneration
import torch

def scrape_news_articles(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h1')
    date = soup.find('span')
    content_div = soup.find('div')
    return {
        "title": title.text.strip() if title else "Title not found",
        "date": date.text.strip() if date else "Date not found",
        "content": content_div.get_text(" ", strip=True) if content_div else "Content not found"
    }

@st.cache_resource
def load_translation_model():
    tokenizer = MarianTokenizer.from_pretrained("fine_tuned_hi_en")
    model = MarianMTModel.from_pretrained("fine_tuned_hi_en")
    return tokenizer, model

@st.cache_resource
def load_summarization_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tokenizer, model

def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def summarize_text(text, tokenizer, model):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def display_news_summary(article_data, translated_title, translated_text, summary):
    st.title("📰 Hindi News Summarizer & Translator")
    st.subheader("📝 Title (Hindi):")
    st.write(article_data['title'])
    st.subheader("🌐 Title (English):")
    st.write(translated_title)
    st.subheader("📅 Published Date:")
    st.write(article_data['date'])
    st.subheader("🗞 Full Article (Hindi):")
    st.write(article_data['content'])
    st.subheader("🌍 Full Article (English):")
    st.write(translated_text)
    st.subheader("🔍 Summary (English):")
    st.markdown(summary)

def main():
    st.sidebar.title("🔗 Hindi News URL Input")
    url = st.sidebar.text_input("Paste a Hindi news article URL:")

    if url:
        try:
            article_data = scrape_news_articles(url)

            if article_data['content'] != "Content not found":
                trans_tokenizer, trans_model = load_translation_model()
                sum_tokenizer, sum_model = load_summarization_model()
                translated_title = translate_text(article_data['title'], trans_tokenizer, trans_model)
                translated_text = translate_text(article_data['content'], trans_tokenizer, trans_model)
                summary = summarize_text(translated_text, sum_tokenizer, sum_model)
                display_news_summary(article_data, translated_title, translated_text, summary)
            else:
                st.error("❌ Could not extract article content.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
