from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from flask import Flask, render_template, request

check_point = "./LaMini-T5-738M"

device = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained(check_point)
model = AutoModelForSeq2SeqLM.from_pretrained(check_point, device_map="auto", torch_dtype=torch.float32, offload_folder="offload")


#@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p = 0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

#@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    db = Chroma(persist_directory='db', embedding_function=embeddings,client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    global question
    question = ""
    answer = None

    if request.method == "POST":
        question = request.form.get("question")
        if question:
            # Get the answer using the question-answering pipeline
            answer, metadata = process_answer(question)
            print(answer)
    return render_template("index.html", answer=answer, question=question)

if __name__ == "__main__":
    app.run(debug=True)