from dotenv import load_dotenv
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQA
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI


st.title("Chatgpt to read Doc files")
text = st.text_input("Enter question you want to ask")

document_directory = "E://testfortext"
loader = DirectoryLoader(document_directory)
documents = loader.load()
# print(documents)
embeddings = OpenAIEmbeddings(openai_api_key="YOUR_API_KEY")

llm = OpenAI(openai_api_key="YOUR_API_KEY")

db = Chroma.from_documents(documents,embeddings)

retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm,retriever=retriever)

    
if st.button("Submit and proceed"):
    generated_text = qa(text)
    st.success(str(generated_text['result']))
