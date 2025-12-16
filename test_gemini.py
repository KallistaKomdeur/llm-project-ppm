import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load variables from .env automatically
load_dotenv()  

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set. Check your .env file and path.")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("models/gemini-2.5-flash")
response = model.generate_content("Say hello in a language of your choice.")
print(response.text)