from model import SemanticSearch
from db import dummy_db, query

# Пример использования
if __name__ == "__main__":

    search = SemanticSearch()
    search.fit(dummy_db)

    duplicates = search.find_duplicates(query)
    print(f'Query: {query["title"]}')
    for idx, score, text in duplicates:
        print(f"Score: {score:.2f} | Title: {text}")


