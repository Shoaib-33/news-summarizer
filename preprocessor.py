
import spacy

nlp = spacy.load('en_core_web_sm')
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