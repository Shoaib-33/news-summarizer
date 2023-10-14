import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
import io
import nltk
from bs4 import BeautifulSoup as soup

from googletrans import Translator
import yake
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gtts import gTTS
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import spacy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
nlp = spacy.load("en_core_web_sm")
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
import joblib
import tensorflow as tf

# Define a custom object scope to register the custom layer

# Load the saved RoBERTa model with the custom object scope

# Now, you can use the loaded_model for inference or further training

nltk.download('punkt')

df = pd.read_csv("train.csv",delimiter=',', encoding='ISO-8859-1')

tweet_df = df[['text','sentiment']]
tweet_df = tweet_df[tweet_df['sentiment'] != 'neutral']

sentiment_label = tweet_df.sentiment.factorize()
tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model=load_model('new.h5')
# history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)
import spacy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from preprocessor import preprocesser

nlp = spacy.load("en_core_web_sm")

text_processer = FunctionTransformer(preprocesser)
import joblib

# Save the trained model to a file

# To load the model back in the future
#define a function for filter stop words and punctuations and extract lemma from the txts
from model import pd

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    return sentiment_label[1][prediction]


# Define the mapping of numerical labels to category names

# Assuming you have already loaded your model as 'loaded_model'
# Load your model here or replace 'loaded_model' with your actual model loading code









# Set Streamlit theme and layout


st.markdown(
    f"""
    <link rel="stylesheet" href="styles.css">
    """,
    unsafe_allow_html=True,
)

def fetch_news_search_topic(topic):
    site = 'https://news.google.com/rss/search?q={}'.format(topic)
    op = urlopen(site)  # Open that site
    rd = op.read()  # read data from site
    op.close()  # close the object
    sp_page = soup(rd, 'xml')  # scrapping data from site
    news_list = sp_page.find_all('item')  # finding news
    return news_list


def fetch_top_news():
    site = 'https://news.google.com/news/rss'
    op = urlopen(site)  # Open that site
    rd = op.read()  # read data from site
    op.close()  # close the object
    sp_page = soup(rd, 'xml')  # scrapping data from site
    news_list = sp_page.find_all('item')  # finding news
    return news_list


def analyze_sentiment_with_model(text):
    # Preprocess the text using the loaded vectorizer
    text_vectorized = vectorizer.transform([text])
    
    # Predict sentiment using the model
    sentiment = model.predict(text_vectorized)
    return sentiment[0]



def fetch_category_news(topic):
    site = 'https://news.google.com/news/rss/headlines/section/topic/{}'.format(topic)
    op = urlopen(site)  # Open that site
    rd = op.read()  # read data from site
    op.close()  # close the object
    sp_page = soup(rd, 'xml')  # scrapping data from site
    news_list = sp_page.find_all('item')  # finding news
    return news_list


def fetch_news_poster(poster_link):
    try:
        u = urlopen(poster_link)
        raw_data = u.read()
        image = Image.open(io.BytesIO(raw_data))
        st.image(image, use_column_width=True)
    except:
        image = Image.open('./Meta/no_image.jpg')
        st.image(image, use_column_width=True)


def display_news_stories(news_list, quantity, target_language=None, enable_audio=False):
    for news in news_list:
        c = 0  # Initialize the counter for each news article
        st.write('**({}) {}**'.format(c, news.title.text))
        news_data = Article(news.link.text)
        try:
            news_data.download()
            news_data.parse()
            news_data.nlp()
        except Exception as e:
            st.error(e)
        
        # Calculate read time estimation
        word_count = len(news_data.text.split())
        read_time_minutes = int(word_count / 200)  # Assuming an average reading speed of 200 words per minute
        
        fetch_news_poster(news_data.top_image)
        
        with st.expander(news.title.text):
            st.markdown(
                '''<h6 style='text-align: justify;'>{}"</h6>'''.format(news_data.summary),
                unsafe_allow_html=True)
            st.markdown("[Read more at {}...]({})".format(news.source.text, news.link.text))
            st.markdown(f"Estimated Read Time: {read_time_minutes} min")
            predicted_sentiment = predict_sentiment(news_data.summary)
            sentiment_emoji = get_sentiment_emoji(predicted_sentiment)
            st.markdown(f"Predicted Sentiment: {sentiment_emoji} ({predicted_sentiment})")
            st.markdown(f"Category of news: {pd(news_data.summary)}")

            if target_language:
                translated_summary = translate_text(news_data.summary, target_language)
                st.markdown("**Translated Summary ({}):**".format(target_language))
                st.markdown(translate_text(news.title.text,target_language))
                st.markdown(translated_summary)
            
          
            
            # Audio Summaries
            if enable_audio:
                audio_summary_button = st.button("Generate Audio Summary")
                if audio_summary_button:
                    audio_path = generate_audio_summary(news_data.summary, lang=target_language)
                    if audio_path:
                        st.audio(audio_path, format='audio/mp3')
                    else:
                        st.warning("Unable to generate audio summary.")
        
        st.success("Published Date: " + news.pubDate.text)
        if c >= quantity:
            break
def generate_audio_summary(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        audio_path = './audio_summary.mp3'
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Error generating audio summary: {e}")
        return None


def analyze_sentiment(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity

    if sentiment_score > 0:
        return "positive"
    elif sentiment_score < 0:
        return "negative"
    else:
        return "neutral"

# Function to get sentiment emoji
def get_sentiment_emoji(sentiment):
    if sentiment == "positive":
        return "😃"
    elif sentiment == "negative":
        return "😞"
    else:
        return "😐"
 
def translate_text(text, target_language):
    try:
        translator = Translator()
        translated_text = translator.translate(text, dest=target_language)
        return translated_text.text
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return ""


def extract_keywords(text):
    custom_kw_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=20)
    keywords = custom_kw_extractor.extract_keywords(text)
    return [kw for kw, _ in keywords]


def run():
    st.title("News Summarizer (In-a-Flash)")
    image = Image.open('./Meta/newspaper.png')

    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write("")

    with col2:
        st.image(image, use_column_width=False)

    with col3:
        st.write("")

    category = ['--Select--', 'Trending🔥 News', 'Favourite💙 Topics', 'Search🔍 Topic']
    cat_op = st.selectbox('Select your Category', category)

    if cat_op == category[0]:
        st.warning('Please select a category!')
    elif cat_op == category[1]:
        st.subheader("✅ Trending🔥 News for you")
        no_of_news = st.number_input('Number of News:', min_value=5, max_value=25, step=1, value=10)
        target_language = st.text_input('Translate to Language (optional):')
        news_list = fetch_top_news()
        display_news_stories(news_list, no_of_news, target_language)
    elif cat_op == category[2]:
        av_topics = ['Choose Topic', 'WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SPORTS', 'SCIENCE', 'HEALTH']
        st.subheader("Choose your favorite Topic")
        chosen_topic = st.selectbox("Choose your favorite Topic", av_topics)
        if chosen_topic == av_topics[0]:
            st.warning("Please choose a topic")
        else:
            no_of_news = st.number_input('Number of News:', min_value=5, max_value=25, step=1, value=10)
            target_language = st.text_input('Translate to Language (optional):')

            news_list = fetch_category_news(chosen_topic)
            if news_list:
                st.subheader(f"✅ Here are some {chosen_topic} News for you")
                display_news_stories(news_list, no_of_news,target_language)
            else:
                st.error(f"No News found for {chosen_topic}")
    elif cat_op == category[3]:
        user_topic = st.text_input("Enter your Topic🔍")
        no_of_news = st.number_input('Number of News:', min_value=5, max_value=15, step=1, value=10)
        target_language = st.text_input('Translate to Language (optional):')


        if st.button("Search", key="search_button") and user_topic:
            user_topic_pr = user_topic.replace(' ', '')
            news_list = fetch_news_search_topic(topic=user_topic_pr)
            if news_list:
                st.subheader(f"✅ Here are some {user_topic.capitalize()} News for you")
                display_news_stories(news_list, no_of_news,target_language)
            else:
                st.error(f"No News found for {user_topic}")

run()