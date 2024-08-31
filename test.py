# import replicate

# # Set your API token here
# replicate_client = replicate.Client(api_token="r8_NGvjLNr9xvndcfRfyWXtR5cf9FV9CYq3mxJzJ")

# input = {
#     "prompt": "no i meant large language model",
#     "max_tokens": 1024
# }

# for event in replicate_client.stream(
#     "meta/meta-llama-3.1-405b-instruct",
#     input=input
# ):
#     print(event, end="")




import replicate
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
import time
import warnings

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

def get_answerllama(query, user_folder, replicate_api_token):
    PERSIST_DIRECTORY = user_folder
    embedding_function = SentenceTransformerEmbedding('sentence-transformers/all-MiniLM-L6-v2')

    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function)

    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    # Retrieve relevant documents
    relevant_docs = retriever.get_relevant_documents(query)

    # Combine the text of the retrieved documents
    combined_docs = " ".join([doc.page_content for doc in relevant_docs])

    # Prepare the prompt for the Replicate API
    input = {
        "prompt": f"Context: {combined_docs}\n\nQuery: {query}\nAnswer:",
        "max_tokens": 1024
    }

    # Make the API call to Replicate
    replicate_client = replicate.Client(api_token=replicate_api_token)
    
    starttime = time.time()
    
    output = replicate_client.run(
        "meta/meta-llama-3.1-405b-instruct",
        input=input
    )
    
    endtime = time.time()
    elapsedtime = endtime - starttime
    print(f'DEBUG : Time taken by Replicate LLaMA: {elapsedtime:.2f} seconds')
    
    print(output)
    return output


replicate_api_token = "r8_doHiDD2F4nXkC9iaXki1TefLV6Wk07q19F9a6"
while True:
    ch = input("Ask your query: ")
    if ch == 'exit':
        break
    else:
        get_answerllama(ch, 'embeddings', replicate_api_token)
