# from openai import OpenAI

# client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# history = [
#     {"role": "system",
#      "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and "
#                 "helpful."},
#     {"role": "user",
#      "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
# ]

# while True:
#     completion = client.chat.completions.create(
#         model="local-model",
#         messages=history,
#         temperature=0.7,
#         stream=True,
#     )

#     new_message = {"role": "assistant", "content": ""}

#     for chunk in completion:
#         if chunk.choices[0].delta.content:
#             print(chunk.choices[0].delta.content, end="", flush=True)
#             new_message["content"] += chunk.choices[0].delta.content

#     history.append(new_message)

#     print()
#     history.append({"role": "user", "content": input("> ")})



from openai import OpenAI
import os

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Read data from the file
data_file_path = "data.txt"
with open(data_file_path, "r") as file:
    data_content = file.read()
    print(data_content)

# Initialize conversation history with data from the file
history = [
    {"role": "system",
     "content": "You are bit predict assistant and respond only according to following content "+ data_content},
    # {"role": "user",
    #  "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
    # {"role": "system",
    #  "content": data_content}  # Add data content to the conversation history
]

while True:
    completion = client.chat.completions.create(
        model="local-model",
        messages=history,
        temperature=0.7,
        stream=True,
    )

    new_message = {"role": "assistant", "content": ""}

    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            new_message["content"] += chunk.choices[0].delta.content

    history.append(new_message)

    print()
    user_input = input("> ")
    history.append({"role": "user", "content": user_input})

    if user_input.lower() == 'exit':
        break
