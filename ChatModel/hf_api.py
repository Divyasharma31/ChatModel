from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

hf_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("please include a api token in .env file")


llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-Nemo-Instruct-2407",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
   
    
)
model = ChatHuggingFace(llm=llm)
result=model.invoke("what is the capital of India?")
print(result.content)

