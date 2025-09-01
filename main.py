import streamlit as st
from vector_store import process_pdf
from agent import build_agent

st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI Agent 🤖")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

if uploaded_file is not None:
    pdf_path = f"./{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.retriever = process_pdf(pdf_path)
    st.session_state.agent = build_agent(st.session_state.retriever)

    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully! You can now chat.")

# Chat interface
if st.session_state.agent:
    st.subheader("💬 Chatbot")
    user_question = st.text_input("Ask a question from the PDF:")

    if user_question.strip() != "":
        response = st.session_state.agent.run(user_question)
        # Save history
        st.session_state.chat_history.append((user_question, response))

    # Show history
    for q, r in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Agent:** {r}")
