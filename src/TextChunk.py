import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextChunker:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def calculate_tfidf_similarity(self, text1, text2):
        tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]

    def split_text_into_sentences(self, text):
        text = str(text)
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences

    def chunk_text(self, docs, retrieve_question):
        sentences = self.split_text_into_sentences(docs)

        chunks = []
        current_chunk = []
        max_similarity = -1

        i = 0
        while i < len(sentences):
            current_chunk.append(sentences[i])
            chunk_text = " ".join(current_chunk)

            similarity = self.calculate_tfidf_similarity(chunk_text, retrieve_question)
            if similarity > max_similarity:
                max_similarity = similarity
                i += 1
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                max_similarity = -1


        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


