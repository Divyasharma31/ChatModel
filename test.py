from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-Nemo-Instruct-2407",
    device=-1 # if GPU available; else remove device argument
)

result = generator("Who is the first president of India?", max_new_tokens=100)
print(result)
