import spacy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
nlp = spacy.load("en_core_web_sm")

#define a function for filter stop words and punctuations and extract lemma from the txts
def preprocesser(text_array):
    preprocessed_texts = []
    for text in text_array:
        doc = nlp(text)
        words_lst = []
        for token in doc:
            if not token.is_stop and not token.is_punct:
                words_lst.append(token.lemma_)
        preprocessed_text = " ".join(words_lst)
        preprocessed_texts.append(preprocessed_text)
    return preprocessed_texts


import joblib

# Save the trained model to a file

# To load the model back in the future
loaded_model = joblib.load('Logisticmainmodel.pkl')


def pd(text):
    label_to_category = {
      0: 'BUSINESS-MONEY',
      1: 'EMPOWERED VOICES',
      2: 'ENVIRONMENT',
       3: 'GENERAL',
       4: 'LIFESTYLE AND WELLNESS',
       5: 'MISC',
       6: 'PARENTING AND EDUCATION',
       7: 'POLITICS',
       8: 'SCIENCE AND TECH',
       9: 'SPORTS AND ENTERTAINMENT',
       10: 'TRAVEL-TOURISM & ART-CULTURE',
       11: 'U.S. NEWS',
       12: 'WORLDNEWS'
      }


    new_texts =[text]
    predicted_labels = loaded_model.predict(new_texts)

# Convert predicted numerical labels to category names using the mapping
    predicted_categories = [label_to_category[label] for label in predicted_labels]

# Print the predicted categories
    for text, predicted_category in zip(new_texts, predicted_categories):
        print(f"Text: {text}\nPredicted Category: {predicted_category}\n")
        return predicted_category



        
pd("election")

