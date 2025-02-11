import streamlit as st
import time
from mistralai import Mistral

st.title("AI ChatBot")



def stream_data(inp):
    for word in inp.split(" "):
        yield word + " "
        time.sleep(0.05)
       
    
if "messages" not in st.session_state:
    st.session_state.messages = []
   

for message in st.session_state.messages:
    i = 1

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    


prompt = st.chat_input("Ask Your Question")


api_key = "dGhT1LuEKYyk8FzaGOQEHhWiLRt5CbVO"
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

if prompt:
    with st.chat_message(""):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"","content":prompt})
    
    with st.spinner("Thinking..."):
        chat_response = client.chat.complete(
                model= model,
                messages=[
                {"role": "user", "content": m["content"]}
                for m in st.session_state.messages
            ],
            
            )
        response = chat_response.choices[0].message.content
        
    with st.chat_message("assistant"):
        
     st.write_stream(stream_data(response))
    st.session_state.messages.append({"role":"assistant", "content":response})
    
    

        
     

    

