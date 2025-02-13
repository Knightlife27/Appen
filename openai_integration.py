import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("API key not found. Please set it in the .env file.")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Function to get a response from OpenAI
def get_openai_response(prompt, model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response for prompt: {prompt}. Error: {e}")
        return None

# Read the CSV file
df = pd.read_csv('prompts.csv')

# Generate responses for each prompt
df['response_1'] = df['prompt'].apply(lambda x: get_openai_response(x, model="gpt-3.5-turbo"))
df['response_2'] = df['prompt'].apply(lambda x: get_openai_response(x, model="gpt-4"))

# Save the updated DataFrame to a new CSV file
df.to_csv('prompts_with_responses.csv', index=False)

print("Responses generated and saved to 'prompts_with_responses.csv'.")
