import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader,PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
chat_history=[]
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=1)
def load_documents(docs_path):
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The path{docs_path}doesn't contain any files")
    documents=[]
    loaders={
        "*.txt":TextLoader,
        "*.pdf":PyPDFLoader,
    }
    for glob_type,loader_type in loaders.items():
        loader=DirectoryLoader(
            path=docs_path,
            glob=glob_type,
            loader_cls=loader_type
        )
        documents.extend(loader.load())
    if len(documents)==0:
        raise FileNotFoundError("Documents are not present")
    return documents
def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks=text_splitter.split_documents(documents)
    return chunks
def store_vector_db(chunks,persist_directory="db/chroma_db"):
    model_name="BAAI/bge-m3"
    model_kwargs={'device': 'cpu'}
    encode_kwargs={'normalize_embeddings': True}
    embedding_model= HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        )
    vectordb=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectordb
def chat(query):
    if chat_history:
        messages=[
            #Constraints or tells the llm what it should do
            SystemMessage(content="Given chat history,rewrite the question to be standalone and just return the rewritten question")
            ]+chat_history+[
            # Human asks the llm
            HumanMessage(content=f"Question:{query}")]
        result=model.invoke(messages)
        question=result.content.strip()
    else:
        question=query
    return question
def ask_ques(vectordb):
    print("Ask llm a question")
    retriever=vectordb.as_retriever(search_kwargs={"k":5})
    while True:
        question=input("Your question")
        if question.lower()=="quit".strip():
            break
        searched_ques=chat(question)
        relevant_docs=retriever.invoke(searched_ques)
        context_text = "\n\n".join([f"- {doc.page_content}" for doc in relevant_docs])
        combined_input = f"""Based on the following documents, please answer this question: {searched_ques}

        Documents:
        {context_text}

        Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer, say "I don't have enough information."
        """
        
        # 4. THE SYNTHESIZER: Send context + history to the LLM
        messages = [
            SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and the previous chat history"),
        ] + chat_history + [
            HumanMessage(content=combined_input)
        ]
        
        result = model.invoke(messages)
        final_answer = result.content
        print(final_answer)

def main():
    docs_path=r"C:\Users\Tharun R Gowda\Desktop\rag\docs"
    persistent_directory="db/chroma_db"
    documents=load_documents(docs_path)
    chunks=split_documents(documents)
    vectordb=store_vector_db(chunks)

    ask_ques(vectordb)
    
if __name__ == "__main__":
    main()
