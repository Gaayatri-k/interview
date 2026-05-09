# app.py

import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


# ==========================================
# LOAD ENV VARIABLES (API KEY HIDDEN)
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="InterviewMentor AI",
    layout="wide"
)

st.title("🎯 InterviewMentor AI – Multi-Round Mock Interview Assistant")


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "score" not in st.session_state:
    st.session_state.score = 0

if "question_count" not in st.session_state:
    st.session_state.question_count = 0


# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================

st.sidebar.header("⚙ Interview Configuration")

interview_round = st.sidebar.selectbox(
    "Select Interview Round",
    [
        "HR Round",
        "Technical Round",
        "System Design Round",
        "Behavioral Round",
        "Aptitude Round"
    ]
)

top_k = st.sidebar.slider(
    "Top-K Retrieval",
    1,
    10,
    4
)

chunk_size = st.sidebar.slider(
    "Chunk Size",
    200,
    2000,
    1000
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    0,
    500,
    200
)


# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_files = st.file_uploader(
    "📄 Upload Resume / Notes / PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embedding_model()


# ==========================================
# LOAD GROQ MODEL
# ==========================================

@st.cache_resource
def load_llm():

    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

llm = load_llm()


# ==========================================
# PROCESS DOCUMENTS
# ==========================================

def process_documents(files):

    all_docs = []

    for file in files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

            tmp_file.write(file.read())
            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)

        docs = loader.load()

        # Add Metadata

        for doc in docs:

            doc.metadata["source"] = file.name

        all_docs.extend(docs)

    # ======================================
    # TEXT SPLITTING
    # ======================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = splitter.split_documents(all_docs)

    # ======================================
    # VECTOR STORE
    # ======================================

    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )

    return vectorstore


# ==========================================
# BUILD QA CHAIN
# ==========================================

def build_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    prompt_template = f"""
You are InterviewMentor AI.

You are conducting a {interview_round} interview.

Rules:
- Answer professionally
- Ask interview-style responses
- Use retrieved context
- If no answer found say:
"I could not find enough information."

Context:
{{context}}

Question:
{{question}}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )

    return qa_chain


# ==========================================
# BUILD KNOWLEDGE BASE
# ==========================================

if st.button("🚀 Build Interview Knowledge Base"):

    if not uploaded_files:

        st.warning("Please upload PDF files")
        st.stop()

    with st.spinner("Processing Documents..."):

        vectorstore = process_documents(uploaded_files)

        qa_chain = build_chain(vectorstore)

        st.session_state.vectorstore = vectorstore
        st.session_state.qa_chain = qa_chain

        st.success("✅ Knowledge Base Ready")


# ==========================================
# INTERVIEW SECTION
# ==========================================

st.header("💬 Mock Interview")

user_question = st.text_input(
    "Ask or Answer Interview Questions"
)


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])


# ==========================================
# QUERY HANDLING
# ==========================================

if user_question:

    if st.session_state.qa_chain is None:

        st.warning("Please build knowledge base first")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):

        st.markdown(user_question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            start_time = time.time()

            result = st.session_state.qa_chain(
                {
                    "query": user_question
                }
            )

            end_time = time.time()

            answer = result["result"]

            source_docs = result["source_documents"]

            retrieval_latency = round(end_time - start_time, 2)

            # ==================================
            # INTERVIEW SCORING
            # ==================================

            score_increment = min(len(user_question.split()) // 5, 10)

            st.session_state.score += score_increment

            st.session_state.question_count += 1

            average_score = round(
                st.session_state.score /
                st.session_state.question_count,
                2
            )

            st.markdown(answer)

            st.success(f"🎯 Interview Score: {average_score}/10")

            # ==================================
            # SOURCE CITATIONS
            # ==================================

            st.subheader("📚 Source Citations")

            for i, doc in enumerate(source_docs):

                source = doc.metadata.get("source", "Unknown")

                page = doc.metadata.get("page", "N/A")

                st.markdown(f"""
### Source {i+1}

- File: {source}
- Page: {page}
""")

                with st.expander(f"View Retrieved Chunk {i+1}"):

                    st.write(doc.page_content)

            # ==================================
            # RETRIEVAL LATENCY
            # ==================================

            st.info(
                f"⏱ Retrieval Latency: {retrieval_latency} seconds"
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ==========================================
# FEATURES SECTION
# ==========================================

st.divider()
