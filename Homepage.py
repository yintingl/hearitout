
import streamlit as st
from streamlit_chat import message
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader, 
    LLMPredictor,
    PromptHelper,
    GPTListIndex,
    GPTTreeIndex
)
from langchain import OpenAI
import openai
import os, logging
import random
import string
import matplotlib.pyplot as plt

api_key = st.text_input('Enter Open AI API Key')
openai.api_key = api_key
#st.write('You entered:', openai.api_key)
OPENAI_API_KEY = api_key

currentPath=os.path.dirname(os.path.realpath(__file__))

def load_or_create_index(filename):
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003",max_tokens=512))
    documents = SimpleDirectoryReader(currentPath+'/tempDir', recursive=True).load_data()
    index = GPTSimpleVectorIndex(documents,llm_predictor=llm_predictor)
    #index.save_to_disk(currentPath+'\Feedback_index.json')
    #index.save_to_disk(currentPath+'\Feedback_list_index.json')
    return index

def save_uploadedfile(uploadedfile):
     with open(os.path.join(currentPath+'/tempDir',uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
        st.success("file uploaded!")
        return uploaded_file.name

def getFeedback(uploaded_file):
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("ISO-8859-1"))
        # st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        # st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file,index_col = 0,encoding='ISO-8859-1')
        st.write(dataframe)
        #return dataframe
    else:
        #st.print("no file")
        st.stop()
        st.rerun()

def query_index(prompt):
    response = index.query(prompt)
    return str(response)
    #return dataDict

def generate_response(prompt):
    completions = openai.Completion.create(
        engine = "text-davinci-003",
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        stop = None,
        temperature=0.5,
    )
    message = completions.choices[0].text
    return message 

def get_text():
    input_text = st.text_input("You: ","Hello, what can I learn about my data?", key="input")
    return input_text
#create a file uploader
#uploaded_file = st.file_uploader("Choose a file")
#show feedback content from uploaded file
#getFeedback(uploaded_file)
#st.set_page_config(layout='wide')


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    uploadedfilename=save_uploadedfile(uploaded_file)
    st.write(uploadedfilename)
    with st.spinner('Preparing summary and Q&A...'):
        index = load_or_create_index(uploadedfilename)
        with st.expander(" Summary"):
            output=query_index("Summarize 5 areas of improvement and explain why")
            st.write(output)
    
    #Creating the chatbot interface
    st.title("ask me anything about your dataset")

    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    user_input = get_text()

    if user_input:
    #output = generate_response(user_input)
        output=query_index(user_input)
    # store the output 
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
        #st.write(st.session_state["generated"][i])
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')








