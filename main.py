
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub # download the prompt from the hub'
from dotenv import load_dotenv

load_dotenv() # load the .env file


if __name__ == '__main__':
    print("Hello World")
    pdf_path = "/home/alcan/repo/langchain/vectorstore-in-memory/2210.03629v3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n") 

    docs = text_splitter.split_documents(documents=documents)

    embedding = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(documents=docs, embedding=embedding)
    vector_store.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings=embedding, allow_dangerous_deserialization=True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    comnine_docs_chain = create_stuff_documents_chain(
        OpenAI(),
        retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(new_vectorstore.as_retriever(), 
                                             comnine_docs_chain)
    
    res = retrieval_chain.invoke({"input": "Give me thet gist of ReAct in 3 sentences"})
    print(res['answer'])