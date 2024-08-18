import numpy as np
import faiss
import itertools
import pickle

class LocalFaissIndex:
    """
    A class representing a local Faiss index for text search.

    Attributes:
        text_encoder: The SentenceTransformer model used for encoding text into vectors.
        index_map: The Faiss index map used for storing and searching vectors.
        faiss_labels: A dictionary mapping vector IDs to corresponding text labels.
        word_hist: A dictionary tracking the frequency of words in the indexed text.
        word_counter: The total count of words in the indexed text.
    """
    def __init__(self):
        self.index_map = faiss.IndexIDMap(faiss.IndexFlatIP(512))
        self.faiss_labels = {}

    def reset(self):
        self.index_map = faiss.IndexIDMap(faiss.IndexFlatIP(512))
        self.faiss_labels = {} 

    def search(self, query_vector, k=1):
        """
        Searches for the nearest vectors to the given query vector.

        Args:
            query (str): The query string.
            k (int, optional): The number of nearest vectors to retrieve. Defaults to 1.

        Returns:
            tuple: A tuple containing the indices, scores, and texts of the nearest vectors.
                - idx (list): The indices of the nearest vectors.
                - scores (list): The scores of the nearest vectors.
                - texts (list): The texts of the nearest vectors.
        """
        query_vector_ = np.array(query_vector)
        if query_vector_.shape[0] > 1:
            query_vector_ = np.expand_dims(query_vector, 0).astype(np.float32)
        faiss.normalize_L2(query_vector_)
        top_k = self.index_map.search(query_vector_, k)
        idx = top_k[1].tolist()[0]
        scores = top_k[0].tolist()[0]
        texts = []
        for id in idx:
            if id != -1:
                text = self.faiss_labels[id]
                texts.append(text)
        return idx, scores, texts

    def insert_normalized(self, text, vector, idx):
        faiss.normalize_L2(vector)
        self.index_map.add_with_ids(vector, idx)
        self.faiss_labels[idx] = text        

    def insert_to_index(self, vector, text):        
        idx = self.index_map.ntotal
        vector_ = np.array(vector)
        faiss.normalize_L2(vector_)
        self._insert(text, vector_, idx)     
        return idx, vector

    def _insert(self, text, vector, idx):
        self.index_map.add_with_ids(vector, idx)
        self.faiss_labels[idx] = text       

    def save(self, db_file_name, labels_file_name):
        faiss.write_index(self.index_map, db_file_name)
        with open(labels_file_name, 'wb') as handle:
            pickle.dump(self.faiss_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, db_file_name, labels_file_name):
        self.index_map = faiss.read_index(db_file_name)
        with open(labels_file_name, 'rb') as handle:
            self.faiss_labels = pickle.load(handle)
        # return(index)

    @staticmethod
    def _take_elem(elem):
        return elem[2]

    @staticmethod
    def _frame_aggregator(triplets):  # Triplet in form ('frame', 'text', 'score')
        triplets = sorted(triplets)
        key_and_group = []

        agg = itertools.groupby(triplets, lambda x: x[0])
        for key, group in agg:
            score = 0
            texts = []
            group_ = list(group)
            for elem in group_:
                texts.append(elem[1])
                score = score + float(elem[2])
            key_and_group.append({key: texts, 'score': score})
            key_and_group = sorted(key_and_group, key=lambda x: x['score'], reverse=True)
            # print({key: list(group_)})
        # print("---------------------------------")
        for kg in key_and_group[:10]:
            print(kg)
