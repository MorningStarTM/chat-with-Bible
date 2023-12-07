from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS

persist_directory = './db'


def main():
    for files in os.listdir('./docs'):
        if files.endswith(".pdf"):
            loader = PyPDFLoader("./docs/The Holy Bible NIV.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=250)
    texts = text_splitter.split_documents(documents)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(texts, embedding_function, persist_directory="./db/", client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

main()