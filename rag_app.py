import os
from dotenv import load_dotenv

load_dotenv() # Load GOOGLE_API_KEY from your .env file

# 1. Load Data
from langchain_community.document_loaders import TextLoader

# Assuming 'data' folder inside your project, and 'article.txt' inside 'data'
# Make sure your current working directory when running the script is the project root
# where 'data' folder resides.
loader = TextLoader("./data/article.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document(s). First 100 chars: {documents[0].page_content[:100]}...")

# 2. Split Data
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunk(s).")

# 3. Embed Data & 4. Store Data (Vector Store)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Ensure you have GOOGLE_API_KEY set in your .env
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Recommended embedding model for Gemini
vectorstore = Chroma.from_documents(chunks, embeddings)
print("Chunks embedded and stored in Chroma vector store.")

# 5. Query & Retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Retrieve top 2 most relevant chunks
print("Retriever created.")

# 6. Augment Prompt & 7. Generate Response
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, convert_system_message_to_human=True)

# Define the prompt that will take the context and question
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based on the provided context ONLY. If you don't know the answer, state that you don't have enough information."),
    ("user", "Context: {context}\n\nQuestion: {input}"),
])

# Create a chain that combines the retrieved documents into the prompt
# This 'stuff' method puts all documents directly into the prompt.
document_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create the full retrieval chain: retriever -> document_chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- Run the RAG Chain ---
print("\n--- Running RAG Chain ---")

# Question 1
question1 = "What is LangChain designed for?"
print(f"\nQuestion: {question1}")
response1 = retrieval_chain.invoke({"input": question1})
print(f"Answer: {response1['answer']}")
# print(f"Source Documents: {response1['context']}") # You can uncomment to see the retrieved chunks

# Question 2
question2 = "What issues does RAG solve?"
print(f"\nQuestion: {question2}")
response2 = retrieval_chain.invoke({"input": question2})
print(f"Answer: {response2['answer']}")

# Question 3 (Outside of provided context)
question3 = "What is the capital of France?"
print(f"\nQuestion: {question3}")
response3 = retrieval_chain.invoke({"input": question3})
print(f"Answer: {response3['answer']}") # Expect it to say it doesn't know or generalize based on prompt