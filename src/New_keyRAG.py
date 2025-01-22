import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_keywords(self, docs):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
        dense_matrix = tfidf_matrix.todense()
        top_keywords = []

        # 提取每个文档的关键词
        for i in range(len(docs)):
            tfidf_scores = dense_matrix[i].tolist()[0]
            sorted_indices = sorted(range(len(tfidf_scores)), key=lambda k: tfidf_scores[k], reverse=True)
            top_keywords_for_doc = [feature_names[idx] for idx in sorted_indices[:10]]

            if len(top_keywords_for_doc) < 10:
                top_keywords_for_doc = [feature_names[idx] for idx in sorted_indices]

            top_keywords.append(top_keywords_for_doc)

        all_keywords = [keyword for sublist in top_keywords for keyword in sublist]


        unique_keywords = list(set(all_keywords))

        if len(unique_keywords) < 10:
            top_keywords_to_return = unique_keywords
        else:
            top_keywords_to_return = unique_keywords[:10]

        # logger.info("To get new keywords")
        return top_keywords_to_return

    def update_query(self, original_query, top_keywords):
        keywords = [keyword for keyword, score in top_keywords]
        updated_query = original_query + " " + " ".join(keywords)
        return updated_query





