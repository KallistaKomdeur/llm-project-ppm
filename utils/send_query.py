from google import genai
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

def get_api_key(provider: str) -> str:
    load_dotenv() 

    key_map = {"gemini": "GEMINI_API_KEY", 
               "openai": "OPENAI_API_KEY",
               "anthropic": "ANTHROPIC_API_KEY"}
    
    if provider not in key_map:
        raise ValueError(f"Unsupported provider: {provider}")
    
    api_key = os.getenv(key_map[provider])
    if not api_key:
        raise RuntimeError(f"API key for {provider} not set")
    
    return api_key

def send_gemini(model_name: str, prompt: str, api_key: str) -> str | None:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=f"gemini-{model_name}",
        contents=prompt,
    )
    return response.text

def send_openai(model_name: str, prompt: str, api_key: str) -> str | None:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def send_anthropic(model_name: str, prompt: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = "".join(block.text for block in response.content if block.type == "text")
    return text

PROVIDERS = {
    "gemini": send_gemini,
    "openai": send_openai,
    "anthropic": send_anthropic,
}

def send_query(provider: str, model_name: str, prompt: str) -> str:
    api_key = get_api_key(provider)
    try:
        return PROVIDERS[provider](model_name, prompt, api_key)
    except KeyError:
        raise ValueError(f"Unsupported provider: {provider}")