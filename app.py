from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st 
import time

vc = FAISS.load_local(
    'vectorstore' , 
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2') , 
    allow_dangerous_deserialization = True
)

model = ChatGroq(
    temperature = 0 ,
    model_name = 'mixtral-8x7b-32768' ,
    groq_api_key = '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')

prompt = ChatPromptTemplate.from_messages(
    [
        (
            'human' ,
            '''
            You are a Constrcurtion expert and your task is to answer queries based on the context provided

            {context}
            Question: {question}
            '''
        )
    ]
)

chain = prompt | model

def answer(query) : 

    similar_docs = vc.similarity_search(
        query ,
        k = 10
    )

    context = ' '.join([
        doc.page_content
        for doc in similar_docs
    ])

    response = chain.invoke({
        'context' : context ,
        'question' : query
    })

    return response.content

def check_prompt(prompt) : 

    '''
    Function to check the prompt

    Args:
    prompt : str : The prompt to be checked

    Returns:
    bool : The boolean value indicating whether the prompt is valid or not
    '''

    try : 
        prompt.replace('' , '')
        return True 
    except : return False


def check_mesaage() : 
    '''
    Function to check the messages
    '''

    if 'messages' not in st.session_state : st.session_state.messages = []

check_mesaage()

for message in st.session_state.messages : 

    with st.chat_message(message['role']) : st.markdown(message['content'])

prompt = st.chat_input('Ask me anything')

if check_prompt(prompt) :

    with st.chat_message('user'): st.markdown(prompt)

    st.session_state.messages.append({
        'role' : 'user' , 
        'content' : prompt
    })

    if prompt != None or prompt != '' : 


        start_time = time.time()
        response = answer(prompt)
        end_time = time.time()

        response = answer(prompt)

        elapsed_time = end_time - start_time

        st.sidebar.write(elapsed_time)

        with st.chat_message('assistant') : st.markdown(response)

        st.session_state.messages.append({
            'role' : 'assistant' , 
            'content' : response
        })