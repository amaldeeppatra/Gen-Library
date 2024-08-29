# import time
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings

 
# def get_answerllama(query, user_folder):
#     ollama = Ollama(
#         base_url='http://localhost:11434',
#         model="llama3"
#     )
   
#     PERSIST_DIRECTORY = user_folder
#     embeddingstype = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
 
#     db = Chroma( persist_directory=PERSIST_DIRECTORY, embedding_function = embeddingstype )
 
#     qa_chain = RetrievalQA.from_chain_type(
#         ollama,
#         retriever=db.as_retriever(search_kwargs={"k":2}),
#     )
#     starttime = time.time()
    
#     return_result = qa_chain.invoke({"query": query})
#     endtime = time.time()
 
#     elapsedtime = endtime - starttime
#     print(f'DEBUG : Time taken by llama3: {elapsedtime:.2f} seconds')
    
#     # print(return_result)
#     print(return_result['result'])
#     return return_result



# # get_answerllama('i want to read harry potter books. but i cant find them', 'embeddings')


# while True:
#     ch = input("Ask your query: ")
#     if ch=='exit':
#         break
#     else:
#         get_answerllama(ch, 'embeddings')



from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import warnings
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class SentenceTransformerEmbedding:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        # Convert to list of lists
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, query):
        # Convert single query embedding to list
        embedding = self.model.encode([query], convert_to_numpy=True)[0]
        return embedding.tolist()

def get_answerllama(query, user_folder):
    ollama = Ollama(
        base_url='http://localhost:11434',
        model="llama3"
    )
   
    PERSIST_DIRECTORY = user_folder
    embedding_function = SentenceTransformerEmbedding('sentence-transformers/all-MiniLM-L6-v2')
 
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function)
 
    qa_chain = RetrievalQA.from_chain_type(
        ollama,
        retriever=db.as_retriever(search_kwargs={"k":2}),
    )
    starttime = time.time()
    
    return_result = qa_chain.invoke({"query": query})
    endtime = time.time()
 
    elapsedtime = endtime - starttime
    print(f'DEBUG : Time taken by llama3: {elapsedtime:.2f} seconds')
    
    print(return_result['result'])
    return return_result

while True:
    ch = input("Ask your query: ")
    if ch == 'exit':
        break
    else:
        get_answerllama(ch, 'embeddings')
