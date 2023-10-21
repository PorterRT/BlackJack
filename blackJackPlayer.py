import openai
import os

# Authenticate with the OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt4_advice(message_content):
    # Convert the message from the file into a message for GPT-4
    message = {
        'role': 'user',
        'content': message_content
    }
    
    # Query the model using the chat interface
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[message]
    )

    # Extract the model's advice from the last message's content
    advice = response.choices[0].message['content'].strip()
    
    return advice

def read_message_from_file(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

# Get message from Player.txt
message_content = read_message_from_file('Player.txt')
print(gpt4_advice(message_content))