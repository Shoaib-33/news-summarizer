# News Summarizer

## Overview:
News Summarizer is an innovative news summarization project designed to provide users with a streamlined and personalized news consumption experience. The platform offers users the ability to access summarized versions of daily news, allowing them to make informed decisions about the articles they wish to explore in greater detail.

## Key Features:

#### Global News Coverage:
Users can stay updated on global affairs by accessing summarized versions of daily news from around the world.

#### Category-Based News Selection:
The platform allows users to tailor their news feed by selecting preferred categories, ensuring a more personalized and relevant news experience.

#### Language Translation:
Users can translate news articles into their preferred language, breaking down language barriers and enabling a more inclusive information consumption experience.

#### Sentiment Analysis:
News Summarizer provides a sentiment analysis feature, categorizing articles as happy, sad, or neutral. This offers users a quick overview of the emotional tone of the news.

#### Category Prediction:
The system predicts the category of each news article and visually presents it to the user. This empowers users to understand the nature of the news they are consuming.

#### Estimated Reading Time:
Users can gauge the time required to read an article, allowing for better time management and planning around their news consumption habits.

#### Link to Full News:
For users interested in delving deeper into a particular news piece, the platform provides a direct link to the full version of the article.

#### Search Functionality:
Users have the ability to search for news on specific topics of interest, and the interface will display relevant articles within the chosen category.

## Data
Dataset was collected from [kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset) which is the "World News Category" Dataset.The dataset has almost 1 lakh 62 thousand data and has 32 categories of features.The dataset was preprocessed and some of the features are grouped which are similar.

## Model 

During the project, experimentation with various models was conducted.Intiallty it was done using LSTM and DistilBert Model.LSTM was not good enough.The DistilBert Model, after hyperparameter tuning, achieved a stable accuracy of 74 percent. Logistic Regression outperformed random models, demonstrating a surprising accuracy of 65 percent. Ultimately, Logistic Regression was selected for its consistent and accurate categorization performance, contributing to an improved output in predicting news categories.


