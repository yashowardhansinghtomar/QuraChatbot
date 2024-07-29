import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import requests

# Title of the Streamlit app
st.title("Q Chatbot")

# URL of the PDF file on GitHub
pdf_url = "https://raw.githubusercontent.com/your-github-username/your-repository/main/your-file.pdf"

# Fetch PDF from URL
response = requests.get(pdf_url)
if response.status_code == 200:
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    # Extract text from PDF
    with fitz.open("temp.pdf") as doc:
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
else:
    st.error("Failed to fetch the PDF file.")
