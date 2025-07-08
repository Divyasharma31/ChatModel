from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents=[
    "Delhi is capital of India",
    "kolata is capita of wst bengal",
    "paris is capital of france"
]
vector=embedding.embed_documents(documents)

print(vector)