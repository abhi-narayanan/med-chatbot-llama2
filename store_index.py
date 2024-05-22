from src.helper import *
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

extracted_data = load_pdf("data/")
text_chunks = text_splitter(extracted_data)

embeddings = download_hugging_face_model()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name = index_name)