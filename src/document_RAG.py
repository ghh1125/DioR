import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentEvaluator:
    def __init__(self, retrieve_question):
        self.retrieve_question = retrieve_question
        self.vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 2))  # TF-IDF with bigrams

    def calculate_cosine_similarity(self, query, doc):
        # print("doc= ",doc)
        if isinstance(doc, dict):
            doc = doc.get('text', '')
        # print("query= ",query)
        # print("doc= ",doc)
        tfidf_matrix = self.vectorizer.fit_transform([query, doc])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity

    def evaluate_retrieved_documents(self, docs):
        # docs is a list of strings, so we can directly use the string
        relevance_scores = [self.calculate_cosine_similarity(self.retrieve_question, doc) for doc in docs]
        # print("Relevance scores:", relevance_scores)
        avg_relevance_score = np.mean(relevance_scores) if relevance_scores else 0
        evaluation_results = {
            'avg_relevance_score': avg_relevance_score,
            'relevance_scores': relevance_scores
        }
        # logger.info("Get new document score")
        return evaluation_results



