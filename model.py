from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.items = None

    def fit(self, products):
        """
        Cоздание эмбеддингов

        :param products: список товаров
        """
        self.products = products
        items = [f"{d['title']} {d['caption']}" for d in products]
        self.embeddings = self.model.encode(items, convert_to_tensor=True)

    def find_duplicates(self, query, threshold=0.8, top_n=5):
        """
        Поиск схожих товаров для заданного запроса

        :param query: текст запроса
        :param threshold: порог косинусного сходства для дубликатов
        :param top_n: количество возвращаемых кандидатов
        :return: список кортежей (индекс, сходство, текст) отсортированных по убыванию сходства
        """
        item = f"{query['title']} {query['caption']}"
        query_embedding = self.model.encode(item, convert_to_tensor=True).unsqueeze(0).cpu().numpy()
        sim_scores = cosine_similarity(query_embedding,
                                           self.embeddings.cpu().numpy()).flatten()

        # Получаем индексы и оценки сходства
        results = [(i, score, self.products[i]['title']) for i, score in enumerate(sim_scores)
                   if score >= threshold]

        # Сортируем по убыванию сходства
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_n]