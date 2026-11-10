from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
client = OpenAI()


import numpy as np

def cosine_similarity(vec1, vec2):
    """
        calculate the cosine similarity between two vectors

        param:
        vec1 = numpy array
        vec2 = numpy array

        returns float (cosine similarity between vec1 vec2) 
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

king_embedding_response = client.embeddings.create(
    input = "king",
    model = "text-embedding-3-large",
)

king_vector = np.array(king_embedding_response.data[0].embedding) # open ai 임베딩 모델에서 추출한 단어 vector를 가져옴
print(f"king vector = {king_vector}")

queun_embedding_response = client.embeddings.create(
    input = "queen",
    model = "text-embedding-3-large",
)

queun_vector = np.array(queun_embedding_response.data[0].embedding)


king_queen_similarity = cosine_similarity(king_vector, queun_vector)
print(f"king queen similarity = {king_queen_similarity}")

tomato_embedding_response = client.embeddings.create(
    input = "tomato",
    model = "text-embedding-3-large",
)
tomato_vector = np.array(tomato_embedding_response.data[0].embedding)

print(f"king tomato similarity = {cosine_similarity(king_vector, tomato_vector)}")

"""
result : 
inho@jeong-inhoui-MacBookPro llm-application % python3 embedding.py
king vector = [ 0.01042938  0.0249939  -0.00149918 ...  0.00830078  0.01049805
 -0.00255775]
king queen similarity = 0.28456225576820254
king tomato similarity = 0.2898436561725781
inho@jeong-inhoui-MacBookPro llm-application % python3 embedding.py
king vector = [ 0.01042938  0.0249939  -0.00149918 ...  0.00830078  0.01049805
 -0.00255775]
king queen similarity = 0.555183063921944
king tomato similarity = 0.2898436561725781
inho@jeong-inhoui-MacBookPro llm-application % 
"""