from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def build_agent(retriever):
    """Create a RetrievalQA agent that answers based on PDF only"""
    llm = OllamaLLM(model="llama3.2", stream=False)

    prompt_template = """
    You are an AI assistant. Answer the question using ONLY the PDF content.
    If the answer is not in the PDF, say:
    "I could not find this information in the document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False,
    )
    return qa_chain
