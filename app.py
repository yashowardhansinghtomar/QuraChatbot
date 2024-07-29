import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Title of the Streamlit app
st.title("PDF Chatbot")

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        pdf_text = ""
        for page in doc:
            pdf_text += page.get_text()
    
    # Display the extracted text (optional, for debugging)
    st.text_area("Extracted Text", pdf_text, height=200)
    
    # Initialize the Hugging Face QA model
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    
    # Input question from the user
    question = st.text_input("Ask a question about the PDF")
    
    if question:
        # Get answer from the model
        result = qa_model(question=question, context=pdf_text)
        st.write("Answer:", result["answer"])
