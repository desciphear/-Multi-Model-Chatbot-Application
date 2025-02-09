
from mistralai import Mistral
from huggingface_hub import InferenceClient

api_key = "dGhT1LuEKYyk8FzaGOQEHhWiLRt5CbVO"
model = "mistral-large-latest"

question = 'quit'

while(True):
    client = Mistral(api_key=api_key)
    question = input("Please Enter a Question: \n")
    question = question.strip().lower()

    if(question == 'quit'):
        break

    if(question != ''):
        chat_response = client.chat.complete(
            model= model,
            messages = [
                {
                    "role": "user",
                    "content": f"{question}",
                },
            ]
        )
        print(chat_response.choices[0].message.content)
    else:
        chat_response = client.chat.complete(
            model= model,
            messages = [
                {
                    "role": "user",
                    "content": "Reply as You do not get the question",          
                },
            ]
        )
        print(chat_response.choices[0].message.content)


