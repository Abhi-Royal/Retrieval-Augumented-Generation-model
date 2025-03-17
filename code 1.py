from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

PINECONE_API_KEY = "your Pinecone key"
HUGGINGFACE_API_KEY = "your Huggingface key"

# Step 1: Load PDF files
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

extracted_data = load_pdf("data/")
print(len(extracted_data))

# Step 2: Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = text_split(extracted_data)
print("Length of my chunk:", len(text_chunks))

# Step 3: Download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()

# Step 4: Initiating Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "question-answer"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Match embedding model output dimensions
        metric="cosine",
        spec=ServerlessSpec(
            cloud="gcp",  # Replace with your cloud provider
            region="us-east-1"  # Replace with your desired region
        )
    )

# Connect to Pinecone index
index = pc.Index(index_name)

# Step 5: Wrap Pinecone Index in a LangChain-Compatible Retriever
vector_store = PineconeVectorStore(index=index, embedding=embeddings) 

# Step 6: Add documents to Pinecone                       ------> If already data is in pinecone database, skip the step-6
for i, chunk in enumerate(text_chunks):
    if chunk.page_content:  # Ensure the chunk has content
        vector = embeddings.embed_query(chunk.page_content)  # Embed the chunk
        metadata = {
            "id": str(i),
            "source": chunk.metadata.get("source", "unknown"),
            "page": chunk.metadata.get("page", -1),
            "text": chunk.page_content,
        }
        index.upsert([(str(i), vector, metadata)])  # Upsert into Pinecone
    else:
        print(f"Skipping empty document chunk at index {i}.")

print("Documents successfully added to Pinecone.")

template = """Use the following retrieved documents to answer the question:
{summaries}
Question: {question}
Answer:"""
PROMPT = PromptTemplate.from_template(template)

# Step 7: Configure HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    task="text-generation",
    temperature=0.8,
    huggingfacehub_api_token=HUGGINGFACE_API_KEY
) 
qa = RetrievalQA.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT}  
)  

# Step 8: Query the Vector Store
while True:
    query = input("Question: ")
    if query.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    # Debugging: Retrieve and print documents from Pinecone     ----> for Debugging
    results = vector_store.as_retriever().get_relevant_documents(query)
    print("Retrieved Documents:", results)

    response = qa.invoke(query)  # Pass input as a dictionary
    print("Response:", response)
