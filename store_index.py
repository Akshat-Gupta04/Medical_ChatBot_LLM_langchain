from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data=load_pdf_file(data='/Users/akshat/Documents/Generative AI/Medical_ChatBot_LLM_langchain/research')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "chatbotmedical"


pc.create_index(
    name=index_name,
    dimension=1536, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = Pinecone.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings)