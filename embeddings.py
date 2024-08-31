from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfminer.high_level import extract_text
import os


def create_embeddings(file_path):
    # Local path to transformers i.e. conversion model
    LOCAL_MODEL_PATH = 'sentence-transformers/all-MiniLM-L6-v2'

    # Create the open-source embedding function
    embeddings = SentenceTransformerEmbeddings(model_name=LOCAL_MODEL_PATH)
    
    # Path of the place to store embeddings
    PERSIST_DIRECTORY = "embeddings" 
    # Check if PERSIST_DIRECTORY directory exists, and create it if not
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)

    # Extract text from PDF
    text = extract_text(file_path)
    page_contents = text
    
    # Create a dictionary to store the document information
    document_info = {
        "document_id": 1,
        "document_name": os.path.basename(file_path),
        "page_content": page_contents,
        "metadata": {
            "source": file_path,
            "page": 1
        }
    }

    data = []
    # Append the document info in the list
    data.append(document_info)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    class DocumentData:
        def __init__(self, page_content, metadata, document_id, document_name):
            self.page_content = page_content
            self.metadata = metadata
            self.document_id = document_id
            self.document_name = document_name

    # Convert document data to DocumentData objects
    all_documents = [DocumentData(**doc) for doc in data]

    # Split documents into chunks using the initialized text splitter
    docs = text_splitter.split_documents(all_documents)

    # Load documents into Chroma
    global db
    db = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIRECTORY)
    db.persist()

    print("DEBUG: Embeddings created successfully")
    return "Embeddings Created"

'''Testing purpose'''
if __name__ == "__main__":
    create_embeddings("books_with_summary.pdf")