
import os
# import Cohere
import cohere

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 

from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain

import streamlit as st

def main():
    """
    This script uses Cohere to generate text based on user input. It prompts the user to enter a Famous Personality's name,
    then find out the birth date. After that this app findout five major events that happened around that time. The
    results are displayed on the Web App's Screen. 
    """
    
    st.title('Correlating Event Search')
    input_text=st.text_input("Enter a Famous Personality's Name")

    input_prompt_1=PromptTemplate(
        input_variables=['name'],
        template="Find out Info Regarding the Personality {name}"
    )

    person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
    dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
    descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

    llm=Cohere(temperature=0.8)
    chain_1=LLMChain(
        llm=llm,prompt=input_prompt_1,verbose=True,output_key='person',memory=person_memory)

    input_prompt_2=PromptTemplate(
        input_variables=['person'],
        template="when was {person} born"
    )

    chain_2=LLMChain(
        llm=llm,prompt=input_prompt_2,verbose=True,output_key='dob',memory=dob_memory)

    # Prompt Templates
    input_prompt_3=PromptTemplate(
        input_variables=['dob'],
        template="Mention 5 major events happened around {dob} in the world"
    )

    chain_3=LLMChain(llm=llm,prompt=input_prompt_3,verbose=True,output_key='description',memory=descr_memory)
    parent_chain=SequentialChain(
        chains=[chain_1,chain_2,chain_3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)

    if input_text:
        st.write(parent_chain({'name':input_text}))

        with st.expander('Person Name'): 
            st.info(person_memory.buffer)

        with st.expander('Major Events'): 
            st.info(descr_memory.buffer)

if __name__ == "__main__":
    main()
