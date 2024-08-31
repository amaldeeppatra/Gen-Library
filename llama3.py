import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import time
import logging
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_answerllama(query, user_folder):
    logger.debug(f'Processing query: {query}')
    ollama = Ollama(
        base_url='http://localhost:11434',
        model="llama3"
    )
   
    PERSIST_DIRECTORY = user_folder
    embeddingstype = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddingstype)

    qa_chain = RetrievalQA.from_chain_type(
        llm=ollama,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
    )
    
    starttime = time.time()
    try:
        result = qa_chain.invoke({"query": query})
    except Exception as e:
        logger.error(f'Error invoking QA chain: {str(e)}')
        raise e
    endtime = time.time()

    elapsedtime = endtime - starttime
    logger.debug(f'Time taken by llama3: {elapsedtime:.2f} seconds')
    
    logger.debug(f'Full result: {result}')
    
    # Adjust based on actual result structure
    if isinstance(result, dict):
        answer = result.get("result") or result.get("answer") or "No answer found."
    elif isinstance(result, str):
        answer = result
    else:
        answer = "No answer found."
    
    logger.debug(f'Extracted answer: {answer}')
    return answer

# Example usage for testing
if __name__ == "__main__":
    answer = get_answerllama('Are there any books by J.K. Rowling and where are they kept?', 'embeddings')
    print(f'Answer: {answer}')
