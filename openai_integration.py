import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import time
import csv

print("Starting the script...")

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("API key not found. Please set it in the .env file.")

print("API key loaded successfully.")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Function to get a response from OpenAI
def get_openai_response(prompt, model="gpt-3.5-turbo"):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Generating response for prompt: {prompt[:50]}... (Model: {model})")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            print("Response generated successfully.")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response for prompt: {prompt[:50]}... Error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(5)
            else:
                print("Max retries reached. Skipping this prompt.")
                return None

# Read the CSV file
print("Reading prompts from 'prompts.csv'...")
df = pd.read_csv('prompts.csv')
print(f"Loaded {len(df)} prompts.")

# Generate responses for each prompt
print("Generating responses using GPT-3.5-turbo...")
df['response_1'] = df['prompt'].apply(lambda x: get_openai_response(x, model="gpt-3.5-turbo"))
print("Finished generating responses with GPT-3.5-turbo.")

print("Generating responses using GPT-4...")
df['response_2'] = df['prompt'].apply(lambda x: get_openai_response(x, model="gpt-4"))
print("Finished generating responses with GPT-4.")

# Save the updated DataFrame to a new CSV file
print("Saving responses to 'prompts_with_responses.csv'...")
df.to_csv('prompts_with_responses.csv', index=False, quoting=csv.QUOTE_ALL, escapechar='\\')

print("Responses generated and saved to 'prompts_with_responses.csv'.")
print("Script execution completed.")
