import cohere
from rich import print
from dotenv import dotenv_values

# Load environment variables
env_vars = dotenv_values(".env")
CohereAPIKey = env_vars.get("CohereAPIKey")

# Initialize Cohere client
co = cohere.Client(api_key=CohereAPIKey)

# Define functions
funcs = [
    "exit", "general", "realtime", "open", "close", "play", "generate image", "system", "content",
    "google search", "youtube search", "reminder"
]

messages = []

preamble = """Your detailed instruction text here..."""

chat_history = [
    {"role": "User", "message": "Hello!"},
    {"role": "Chatbot", "message": "Hi! How can I assist you today?"},
    {"role": "User", "message": "What is Python?"},
    {"role": "Chatbot", "message": "Python is a popular programming language..."},
]

def FirstLayerDMM(prompt: str = "test"):
    messages.append({"role": "User", "message": f"{prompt}"})

    stream = co.chat_stream(
        model='command-r-plus',
        message=prompt,
        temperature=0.7,
        chat_history=chat_history,
        prompt_truncation='OFF',
        connectors=[],
        preamble=preamble
    )

    response = ""

    for event in stream:
        if event.event_type == "text-generation":
            response += event.text

    response = response.replace("\n", "").split(",")
    response = [i.strip() for i in response]

    temp = []
    for task in response:
        for func in funcs:
            temp.append(task)

    response = temp

    if "(query)" in response:
        return FirstLayerDMM(prompt=prompt)
    else:
        return response

if __name__ == "__main__":
    while True:
        print(FirstLayerDMM(input(">>> ")))
