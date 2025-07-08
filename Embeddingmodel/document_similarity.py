from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
print("Script started")


model=SentenceTransformer('all-MiniLM-L6-v2')

document=[
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query ='tell me about dhoni'

doc_embeddings=model.encode(document)
# [querry] because model expects a list....and 0 as we need the very first one
query_embeddings=model.encode([query])[0]

score=cosine_similarity([query_embeddings],doc_embeddings)[0]
# x[0] = index (like 0, 1, 2),,,,,x[1] = score (like 0.1, 0.4, 0.9),,,[-1]for the last item
# sorting based on score rather than index,index id=s just to tell aat which part it is present
index,score=sorted(list(enumerate(score)),key=lambda x:x[1])[-1]

print("Query",query)
print("Best match",document[index])
print("Similarity Score",score)