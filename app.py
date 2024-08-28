from flask import Flask, request, jsonify, render_template
import time
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

app = Flask(__name__)

def get_answerllama(query, user_folder):
    ollama = Ollama(
        base_url='http://localhost:11434',
        model="llama3"
    )
   
    PERSIST_DIRECTORY = user_folder
    embeddingstype = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
 
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddingstype)
 
    qa_chain = RetrievalQA.from_chain_type(
        ollama,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
    )
    starttime = time.time()
    
    return_result = qa_chain.invoke({"query": query})
    endtime = time.time()
 
    elapsedtime = endtime - starttime
    print(f'DEBUG: Time taken by llama3: {elapsedtime:.2f} seconds')
    
    return return_result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "pong"}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    query = data.get('query')
    user_folder = "embeddings"  # Update with the appropriate directory path
    result = get_answerllama(query, user_folder)
    
    return jsonify({"result": result['result']}), 200

if __name__ == '__main__':
    app.run(debug=True)
