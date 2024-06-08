from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import numpy as np


class KnnSearch:
    def __init__(self, data=None, num_trees=None, emb_dim=None):
        self.num_trees = num_trees
        self.emb_dim = emb_dim

    def get_embeddings_for_data(self, data_ls):
        # NOTE: any embedding model from sentence-transformers can be used
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embeddings = model.encode(data_ls)
        return embeddings

    def get_top_n_neighbours(self, sentence, data_emb, transfer_data, k):
        """
        Retrieves the top k most similar questions for "sentence" based on cosine similarity from given embeddings "data_emb".

        Parameters:
        sentence (str): The input sentence to find similar questions for.
        data_emb (np.ndarray): The embeddings for the transfer questions.
        transfer_data (list): The list of transfer questions corresponding to data_emb.
        k (int): The number of top similar questions to retrieve.

        Returns:
        list: A list of the top k similar questions from transfer_data and all similar questions from str_qa.
        """
        sent_emb = self.get_embeddings_for_data(sentence)
        top_questions = []

        text_sims = cosine_similarity(data_emb, [sent_emb]).tolist()
        results_sims = zip(range(len(text_sims)), text_sims)
        sorted_similarities = sorted(results_sims, key=lambda x: x[1], reverse=True)

        for idx, item in sorted_similarities[:k]:
            top_questions.append(transfer_data[idx])

        return top_questions

    def semantic_score(self, question, document):
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        query_embedding = model.encode(question)
        document_embeddings = model.encode(document)

        return np.dot(query_embedding, document_embeddings)

    def semantic_values(self, questions, documents):
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        query_embedding = model.encode(questions)
        document_embeddings = model.encode(documents)

        semantic_scores = []
        for question_emb in query_embedding:
            for document_emb in document_embeddings:
                semantic_scores.append(self.semantic_score(question_emb, document_emb))

        return np.mean(np.array(semantic_scores)), np.std(np.array(semantic_scores))

    def temporal_score(self, sentence_timestamp: datetime, data_emb_timestamp: datetime):
        alpha = 1.0
        return alpha / (data_emb_timestamp.timestamp() - sentence_timestamp.timestamp())

    def get_top_n_temp_neighbours(self, sentence_timestamp: datetime, data_emb_timestamps: list[datetime],
                                  standard_deviation_sematic, mean_sematic, transfer_data, k):
        """
        Retrieves the top k most similar questions for "sentence" based on cosine similarity from given embeddings "data_emb".

        Parameters:
        sentence (str): The input sentence to find similar questions for.
        data_emb (np.ndarray): The embeddings for the transfer questions.
        transfer_data (list): The list of transfer questions corresponding to data_emb.
        k (int): The number of top similar questions to retrieve.

        Returns:
        list: A list of the top k similar questions from transfer_data and all similar questions from str_qa.
        """
        timestamps = np.array([dt.timestamp() for dt in data_emb_timestamps])
        mean_timestamp = np.mean(timestamps)
        standard_deviation_timestamp = np.std(timestamps)

        temp_scores = []
        for data_emb_timestamp in data_emb_timestamps:
            temp_score = ((self.temporal_score(sentence_timestamp,
                                               data_emb_timestamp) - mean_timestamp) / standard_deviation_timestamp) * standard_deviation_sematic + mean_sematic
            temp_scores.append(temp_score)

        results_sims = zip(range(len(temp_scores)), temp_scores)
        sorted_similarities = sorted(results_sims, key=lambda x: x[1], reverse=True)

        top_questions = []
        for idx, item in sorted_similarities[:k]:
            top_questions.append(transfer_data[idx])

        return top_questions


if __name__ == '__main__':
    # Load the model
    # TODO: use sematic_score
    # semantic_score("what's your name?", "His favourite food is pizza")

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # Semantic search
    documents = ["The cat sits outside.", "A man is playing guitar.", "The new movie is awesome.", "I love pasta."]
    document_embeddings = model.encode(documents)
    query = "I like spaghetti."
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], document_embeddings)
    most_similar_idx = np.argmax(similarities)
    print(f"Most similar document: {documents[most_similar_idx]}")

    # Paraphrase mining
    sentences = [
        "The cat sits outside.",
        "A man is playing guitar.",
        "The new movie is awesome.",
        "The cat is sitting outside.",
        "The guitar is being played by a man."
    ]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    paraphrases = util.paraphrase_mining(model, sentences)

    for paraphrase in paraphrases:
        score, i, j = paraphrase
        print(f"Score: {score:.4f} - Sentence 1: {sentences[i]} - Sentence 2: {sentences[j]}")
