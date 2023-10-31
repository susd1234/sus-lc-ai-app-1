
import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain

import streamlit as st

def main():
    """
    This script uses OpenAI's GPT-3 to generate text based on user input. It prompts the user to enter a celebrity's name,
    then uses GPT-3 to generate the celebrity's birth date and five major events that happened around that time. The
    results are displayed in a Streamlit app.
    """
    # streamlit framework
    st.title('Personality Search Results')
    input_text=st.text_input("Search the topic u want")

    # Prompt Templates
    first_input_prompt=PromptTemplate(
        input_variables=['name'],
        template="Tell me about celebrity {name}"
    )

    # Memory
    person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
    dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
    descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

    ## OPENAI LLMS
    llm=OpenAI(temperature=0.8)
    chain=LLMChain(
        llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)

    # Prompt Templates
    second_input_prompt=PromptTemplate(
        input_variables=['person'],
        template="when was {person} born"
    )

    chain2=LLMChain(
        llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)

    # Prompt Templates
    third_input_prompt=PromptTemplate(
        input_variables=['dob'],
        template="Mention 5 major events happened around {dob} in the world"
    )

    chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=descr_memory)
    parent_chain=SequentialChain(
        chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'],verbose=True)

    if input_text:
        st.write(parent_chain({'name':input_text}))

        with st.expander('Person Name'): 
            st.info(person_memory.buffer)

        with st.expander('Major Events'): 
            st.info(descr_memory.buffer)

if __name__ == "__main__":
    main()
