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

df = pd.read_csv("dataset/train.csv",delimiter=',', encoding='ISO-8859-1')

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
model=load_model('models/new.h5')
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
        image = Image.open('./picture/no_image.jpg')
        st.image(image, use_column_width=True)


def display_news_stories(news_list, quantity, target_language=None, enable_audio=False):
    for news in news_list:
        c = 0  # Initialize the counter for each news article
        st.write('**<span style="color: #f0f0f0;">({}) {}</span>**'.format(c, news.title.text), unsafe_allow_html=True)

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
    '''<h6 style='text-align: justify; color: #f0f0f0; font-weight: bold;'>{}</h6>'''.format(news_data.summary),
            unsafe_allow_html=True)
            st.markdown("[Read more at {}...]({})".format(news.source.text, news.link.text))
            st.markdown("<span style='color:#ffffff;'>Estimated Read Time: {} min</span>".format(read_time_minutes), unsafe_allow_html=True)
            predicted_sentiment = predict_sentiment(news_data.summary)
            sentiment_emoji = get_sentiment_emoji(predicted_sentiment)
            st.markdown("<span style='color: #ffffff;'>Predicted Sentiment: {} ({})</span>".format(sentiment_emoji, predicted_sentiment), unsafe_allow_html=True)

# Set the Category of news text with custom style
            st.markdown("<span style='color: #ffffff;'>Category of news: {}</span>".format(pd(news_data.summary)), unsafe_allow_html=True)
            if target_language:
                translated_summary = translate_text(news_data.summary, target_language)
                st.markdown("<span style='color: #ffffff; font-weight: bold;'>Translated Summary ({})</span>:".format(target_language), unsafe_allow_html=True)
                news_title_translated = translate_text(news.title.text, target_language)

# Set the translated text with custom style
                st.markdown("<span style='color: #ffffff;'>{}</span>".format(news_title_translated), unsafe_allow_html=True)
                st.markdown("<span style='color: #ffffff;'>{}</span>".format(translated_summary), unsafe_allow_html=True)
            
          
            
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
    

# Define a custom CSS class to change the background color of the Streamlit app
    custom_css ="""
<style>
    .stApp {
        background-image: url('https://img.freepik.com/free-vector/global-technology-earth-news-bulletin-background_1017-33687.jpg?w=1380&t=st=1697978148~exp=1697978748~hmac=4943a05997b7d4461e9e581e177b3a5dcca3df44d6fa519f830ebe1b922fcfa0'); /* Replace with your image file name */
        background-color: #333; /* Fallback color if the image is unavailable */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center center;
        opacity: 0.9;
    }
</style>
"""

# Display the custom CSS using st.markdown
    st.markdown(custom_css, unsafe_allow_html=True)

# Your Streamlit app content goes here



# Define a custom CSS class with styles for the centered header

# Define a custom CSS class with styles for the centered header
    custom_css = """
<style>
    .custom-header {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 20;
        text-align: center;
        color: #002366;
        background: rgba(245, 245, 245, 0.7); /* Transparent whitish background */
        border: 2px solid #0074D9; /* Stylish border color */
        border-radius: 15px; /* Circular border radius for a stylish look */
        font-family: 'Bebas Neue', sans-serif;
        font-size: 60px;
        text-transform: uppercase;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.3); /* Box shadow for depth and style */
        
    }
    .sub-header {
        font-size: 25px;
        color: #f0f0f0; /* Set font color to white */
        text-align: center; /* Center the text */
        margin-left: 20px;
    }
</style>

"""

# Display the custom CSS using st.markdown
    st.markdown(custom_css, unsafe_allow_html=True)

# Use the custom class on your centered header element
    st.markdown("<div class='custom-header'>NewsWaves</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>A platform to get daily latest news updates of your favorite category.</div>", unsafe_allow_html=True)
# The rest of your Streamlit app goes here


# The rest of your Streamlit app goes here

    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write("")
        

    

    with col3:
        st.write("")

    category = ['Select any category', 'Latest News', 'Favourite News', 'Search Any News']
    cat_op = st.selectbox('Select your Category', category)

    if cat_op == category[0]:
        st.warning('Please select a category!')
    elif cat_op == category[1]:
        st.markdown("<h3 style='color: #ffffff; font-weight: bold;'>Latest News for you</h3>", unsafe_allow_html=True)
        st.markdown("<span style='color: #ffffff;'>Number of News:</span>", unsafe_allow_html=True)

# Set the number of news input
       

# Set the number of news input with custom style for deep black font
        no_of_news = st.number_input('', min_value=5, max_value=25, step=1, value=10, format="%d", key="no_of_news")
        st.markdown("<style>div[data-baseweb='input'] input { color: #000000 !important; }</style>", unsafe_allow_html=True)

        st.markdown("<span style='color: #ffffff;'>Translate to Language (optional):</span>", unsafe_allow_html=True)

# Set the target language input with reduced newline
        target_language = st.text_input('', key="target_language")
        st.markdown("<style>div[data-baseweb='input'] input { margin-top: 0; color: #ffffff; }</style>", unsafe_allow_html=True)
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

st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #333;
            color: white;
            padding: 10px;
            text-align: center;
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display your contact information in the footer bar
st.markdown(
    """
    <div class="footer">
        Developed by:-Md Shoaib Shahriar Ibrahim | shoaibshahriar@iut-dhaka.edu | [GitHub Profile](https://github.com/Shoaib-33)
    </div>
    """,
    unsafe_allow_html=True
)

run()
