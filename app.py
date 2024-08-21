from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import streamlit as st
import pandas as pd
import io
import time
import openai



def create_embeddings():
    Hitachi_path = "./hitachi_files"
    NetApp_path = "./NetApp_files"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )

    ##Creating embeddings for Hitachi Documents
    loader = PyPDFDirectoryLoader(Hitachi_path)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    # print(texts)
    Chroma.from_documents(documents=texts, 
                        embedding=embeddings,
                        collection_name="Hitachi_Collection", 
                        persist_directory=r"./chroma",
                        collection_metadata={"hnsw:space": "cosine"} )


    ##Creating Embeddings for NetApp Documents
    loader = PyPDFDirectoryLoader(NetApp_path)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    # print(texts)
    Chroma.from_documents(documents=texts, 
                        embedding=embeddings,
                        collection_name="NetApp_Collection", 
                        persist_directory=r"./chroma",
                        collection_metadata={"hnsw:space": "cosine"} )


def ask_question(query,collection_name):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=0)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists("./chroma"):
        db = Chroma(persist_directory=r"./chroma", 
                    collection_name=collection_name, 
                    embedding_function=embeddings)
    else:
        st.write("Creating the Embedding File")
        start_time = time.time()
        create_embeddings()
        end_time = time.time()
        st.write(f"Time taken to create the embeddings: {(end_time-start_time)/60}Mins")
        db = Chroma(persist_directory=r"./chroma", 
                    collection_name=collection_name, 
                    embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={'k':10})

    system_prompt = (
        """You are an AI agent designed to perform question-answering task over the various documents related to the customer queries. You will be provided with the context and the user question. Your task is to carefully analyze the given context and answer the users question in clear, detailed and well format structure. Below are the few important points that you need to remember:
         - If there is no answer available; the output should be “No Answer”
         - If partial answer is available; The output should include “Partial Answer Available”
         - Answer length should be not exceed more than 75-80 words or 7-8 lines. This is very important, if needed summarize the answer to fit within this range.
         - You are strictly not allowed to use your own knowledge base to answer any of the user questions
         - You are allowed to answer from the provided context only
         - It may happen that the provided context may contain some information that is not retaled with the user questions. In this case you need to ignore the irrelevant contents and use only relevent content to give the final output to the user.
         - If user requested information is not present in the given context simply say 'No Answer' without any extra words or expalanation. Don't try to makeup the answer using your own knowledge
         - Remeber you are not allowed to add any extra words apart from the final output. Don't say anything like 'Based on the provided context', Simply give your final answer with any addition of extra words.
         - Remember to carefully analyze the complete context before giving the final output.
         - Remember if the answer is exceeding the given limit of 90-110 words or 10-12  lines, you need to summarise the answer to fit within the limit by keeping the key points and then provide your final answer.

        Special Consideration:
        - Provide the detailed description of proposed solution and this question is mandatory. If it is exceeding the limit of 75-80 words, summarize it within the given limit and give the final answer.
        - RESTFul API and REST API are Same
        Below is the context:
        """
        "\n\n"
        "{context}"
    )
    # system_prompt = (
    #     """You are an AI agent desined to perform question-answering task over the various documents related to the customer Queries. You will be provided with the context and the user question. Your task is to carefully analyze the given context and answer the users question in clear, detailed and well format structure. You are having expertise in answering the questions within the range of 75-80 words or 7-8 lines. Below are the few important points that you need to remember:
    #      - You first analyze all context given to you before answering any questions.
    #      - If there is no answer available in the context for the given queries, simply say "No Answer" without any extra word as your final output. Don't try to makeup the answer using your own knowledge base.
    #      - If Partial Answer is available; The output should include “Partial Answer Available”
    #      - You are strictly not allowed to use your own knowledge base to answer any of the user questions
    #     Below is the context:
    #     """
    #     "\n\n"
    #     "{context}"
    # )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input":query})
    sources = []

    for doc in response["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_number": doc.metadata["page"], "page_content": doc.page_content}
        )
    references_dict = {
        "references": {}
    }

    for i, ref in enumerate(sources):
        ref_key = f"reference {i + 1}."
        references_dict["references"][ref_key] = {
            "doc_name": ref['source'],
            "page_number": ref['page_number'],
            "content": ref['page_content']
        }
    output = {
        "answer": response['answer'],
        "references": references_dict['references']
    }
    return output


def main():
    tabs = st.tabs(["Hitachi App", "NetApp"])
    with tabs[0]:
        st.title("RFP RESPONSE HACKATHON: Hitachi App")
        st.markdown("This demo showcases document embedding and retrieval using GROQ API REFERENCE for Hitachi.")

        uploaded_file = st.file_uploader("Upload an Excel file with questions", type=["xlsx"], key="hitachi")
        if st.button("Get Answers",key="Hitachi_Button"):
            
            if uploaded_file is not None:
                questions_df = pd.read_excel(uploaded_file)
                questions = questions_df['Sample RFP Questions asked by Customer'].tolist()  
            else:
                st.warning("Please upload an Excel file containing the questions.")
                questions = []  # No questions to process

            if questions:
                responses = []
                for question in questions:
                    response = ask_question(question,"Hitachi_Collection")
                    answer = response['answer']
                    responses.append({'Question': question, 'Answer': answer})
                    
                df = pd.DataFrame(responses)
                st.write(df)    

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Sheet1')   

                st.download_button(
                    label="Download Responses as Excel",
                    data=output.getvalue(),
                    file_name='Hitachi_responses.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

                with st.expander("Document Similarity Search"):
                    for doc in responses:
                        st.write(doc['Answer'])
                        st.write("--------------------------------")



    with tabs[1]:
        st.title("RFP RESPONSE HACKATHON : NetApp")
        st.markdown("This demo showcases document embedding and retrieval using GROQ API REFERENCE for NetApp.")
        uploaded_file = st.file_uploader("Upload an Excel file with questions", type=["xlsx"], key="netapp")
        if st.button("Get Answers",key="NetApp_Button"):
            if uploaded_file is not None:
                questions_df = pd.read_excel(uploaded_file)
                questions = questions_df['Sample RFP Questions asked by Customer'].tolist()  
            else:
                st.warning("Please upload an Excel file containing the questions.")
                questions = []  # No questions to process

            if questions:
                responses = []
                for question in questions:
                    response = ask_question(question,"NetApp_Collection")
                    answer = response['answer']
                    responses.append({'Question': question, 'Answer': answer})

                df = pd.DataFrame(responses)
                st.write(df)    

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Sheet1')   

                st.download_button(
                    label="Download Responses as Excel",
                    data=output.getvalue(),
                    file_name='NetApp_responses.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

                with st.expander("Document Similarity Search"):
                    for doc in responses:
                        st.write(doc['Answer'])
                        st.write("--------------------------------")

if __name__ == '__main__':
    main()