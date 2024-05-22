from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_model
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from src.prompt import *
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from the .env file
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings = download_hugging_face_model()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(embedding=embeddings, index_name = index_name)

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
chain_type_kwargs={"prompt":prompt}

llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever= docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents= True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template("chat.html")

if __name__ == "__main__":
    app.run(debug=True)