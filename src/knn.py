from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

class KnnSearch:
    def __init__(self, data=None, num_trees=None, emb_dim=None):
        self.num_trees = num_trees
        self.emb_dim = emb_dim

    def get_embeddings_for_data(self, data_ls, transformer='sentence-transformers/all-mpnet-base-v2', convert_to_tensor=False):
        # NOTE: any embedding model from sentence-transformers can be used
        model = SentenceTransformer(transformer)
        embeddings = model.encode(data_ls, convert_to_tensor=convert_to_tensor)
        return embeddings

    def get_top_n_neighbours1(self, sent_emb, data_emb, transfer_data, k):
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
        hits = util.semantic_search(sent_emb, data_emb, top_k=k)[0]
        top_k_question_ids = [hit['corpus_id'] for hit in hits]

        return [transfer_data[id] for id in top_k_question_ids]

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

        text_sims = cosine_similarity(data_emb.cpu().numpy(),[sent_emb]).tolist()
        results_sims = zip(range(len(text_sims)), text_sims)
        sorted_similarities = sorted(results_sims, key=lambda x: x[1], reverse=True)  # Obtain the highest similarities

        # NOTE: we only match based on questions, but include the full question-answer pair in resulting neighs
        for idx, item in sorted_similarities[:int(k)]:
            top_questions.append(transfer_data[idx])

        return top_questions