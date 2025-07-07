from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os

os.environ['HF-HOME']='D:/huggingface_cache'

llm=HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mistral-Nemo-Instruct-2407",
    task="text-generation",
    model_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model=ChatHuggingFace(llm=llm)

result=model.invoke("who is the first president of india")

print(result.content)