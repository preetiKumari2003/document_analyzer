import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.schema import Document
import tiktoken  # For token estimation
import os
from pinecone import Pinecone, ServerlessSpec


import streamlit as st

# Set environment variables
os.environ["PINECONE_API_KEY"] = "pcsk_3MBdvm_9oPCXJzkj9gGe9pR9MdHNS8EH2pkgB2L6YLRv3qo1SMDVwK9NzFfdKVSxuRLzg3"
os.environ["OPENAI_API_KEY"] = "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"  # Replace with your OpenAI API key

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "analyzer"
dimension = 1536
namespace = "uploaded_files"

# Create Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)


def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

# App Title
st.title("Document Content Analyzer")
st.sidebar.header("Uploaded Files")

# Retrieve previously uploaded files
uploaded_file_names = []
search_results = index.query(vector=[0] * dimension, namespace=namespace, top_k=1000, include_metadata=True)
matches = search_results.get("matches", [])
for match in matches:
    file_name = match.get("metadata", {}).get("file_name", "")
    if file_name and file_name not in uploaded_file_names:
        uploaded_file_names.append(file_name)

# File uploader
uploaded_files = st.file_uploader("Upload PDF/Text files", type=["pdf", "txt"], accept_multiple_files=True)

# Handle newly uploaded files
if uploaded_files:
    for file in uploaded_files:
        if file.name not in uploaded_file_names:
            # Extract content and store embeddings
            raw_text = ""
            if file.name.endswith(".pdf"):
                reader = PdfReader(file)
                for page in reader.pages:
                    content = page.extract_text()
                    if content:
                        raw_text += content
            elif file.name.endswith(".txt"):
                raw_text += file.read().decode("utf-8")

            if raw_text:
                text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100, length_function=len)
                texts = text_splitter.split_text(raw_text)
                embeddings = OpenAIEmbeddings()

                vectors = []
                for i, text in enumerate(texts):
                    embedding = embeddings.embed_query(text)
                    vectors.append((f"{file.name}-{i}", embedding, {"text": text, "file_name": file.name}))

                if vectors:
                    index.upsert(vectors=vectors, namespace=namespace)
                    uploaded_file_names.append(file.name)
                    st.success(f"Embeddings for '{file.name}' stored successfully!")

# Dropdown for file selection
selected_files = st.sidebar.multiselect("Select files to analyze", uploaded_file_names)

# Query input
query = st.text_input("Enter your query:")
if st.button("Send Query"):
    if query:
        st.info("Processing query...")

        # Filter matches based on selected files
        filtered_matches = [
            match for match in matches if match.get("metadata", {}).get("file_name") in selected_files
        ]

        if filtered_matches:
            # Combine the text chunks while keeping within token limits
            max_tokens = 3500  # Leave room for the query and completion
            current_tokens = 0
            selected_documents = []

            for match in filtered_matches:
                text = match.get("metadata", {}).get("text", "")
                tokens = count_tokens(text)
                if current_tokens + tokens <= max_tokens:
                    selected_documents.append(Document(page_content=text, metadata={"source": match.get("id")}))
                    current_tokens += tokens
                else:
                    break

            if selected_documents:
                # Run the query against the filtered documents
                chain = load_qa_chain(OpenAI(), chain_type="stuff")
                answer = chain.run(input_documents=selected_documents, question=query)
                st.write(f"**Answer:** {answer}")
            else:
                st.warning("Selected documents exceed the token limit. Please refine your selection.")
        else:
            st.warning("No content available for the selected files. Please ensure files are uploaded and selected.")

        # Save chat history
        with open("chat_history.txt", "a") as f:
            f.write(f"Q: {query}\nA: {answer}\n")

# Display persistent chat history
if "chat_history.txt" in os.listdir():
    with open("chat_history.txt", "r") as f:
        chat_history = f.read()
        st.text_area("Chat History", value=chat_history, height=200, disabled=True)
