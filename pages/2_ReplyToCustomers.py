
import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

import openai
#openai.api_key = os.getenv("OPENAI_API_KEY")
api_key = st.text_input('Enter Open AI API Key')
openai.api_key = api_key
st.markdown("# Respond to Customer Feedback with AI Assistance")
start_phrase = 'Write a polite response to the following customer feedback'

currentPath=os.getcwd()

with st.form(key="form_email"):
    start = st.text_input("Enter customer feedback for an auto-generated response:")
    st.write(f"(Example: The menu settings are all over the place and it's absolutely ridiculous.)")

    slider = st.slider("How many characters do you want your email to be? ", min_value=64, max_value=750)
    st.write("(A typical email is usually 100-500 characters)")

    submit_button = st.form_submit_button(label='Generate Replies')

    if submit_button:
        with st.spinner("Generating Email..."):
            output = openai.Completion.create(model="text-davinci-003", prompt=start_phrase+start, max_tokens=slider)
            text = output['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
        st.markdown("# Email Output:")
        st.write(text)
        #print(start_phrase+text)



        st.markdown("____")
        st.markdown("You can press the Generate Email Button again if you're unhappy with the model's output")
        #st.form_submit_button(label='Send')
        
        #st.subheader("Otherwise:")
        #st.text(output)
        #url = "https://mail.google.com/mail/?view=cm&fs=1&to=&su=&body=" + backend.replace_spaces_with_pluses(
           # start + output)

        #st.markdown("[Click me to send the email]({})".format(url))

#print uploaded texts - prove if a file upload is successful
# def getFeedback(uploaded_file):
#     if uploaded_file is not None:
#         # To read file as bytes:
#         bytes_data = uploaded_file.getvalue()
#         # st.write(bytes_data)

#         # To convert to a string based IO:
#         stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#         # st.write(stringio)

#         # To read file as string:
#         string_data = stringio.read()
#         # st.write(string_data)

#         # Can be used wherever a "file-like" object is accepted:
#         dataframe = pd.read_csv(uploaded_file,index_col = 0)
#         dataframe.rename(columns={'prompt': 'Feedback'},inplace=True)
#         st.write(dataframe[['Feature','Feedback']])
#         #return dataframe
#     else:
#         #st.print("no file")
#         st.stop()
#       st.rerun()

#define title of the landing page
