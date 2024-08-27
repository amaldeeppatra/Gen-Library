import time
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

 
def get_answerllama(query, user_folder):
    ollama = Ollama(
        base_url='http://localhost:11434',
        model="llama3"
    )
   
    PERSIST_DIRECTORY = user_folder
    embeddingstype = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
 
    db = Chroma( persist_directory=PERSIST_DIRECTORY, embedding_function = embeddingstype )
 
    qa_chain = RetrievalQA.from_chain_type(
        ollama,
        retriever=db.as_retriever(search_kwargs={"k":2}),
    )
    starttime = time.time()
    
    return_result = qa_chain.invoke({"query": query})
    endtime = time.time()
 
    elapsedtime = endtime - starttime
    print(f'DEBUG : Time taken by llama3: {elapsedtime:.2f} seconds')
    
    print(return_result)
    return return_result



get_answerllama('are there any books by cs lewis', 'embeddings')