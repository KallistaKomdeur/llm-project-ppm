from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic

def get_api_key(provider):
    load_dotenv()
    key_map = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    if provider not in key_map:
        raise ValueError(f"Unknown provider: {provider}")
    api_key = os.getenv(key_map[provider])
    if not api_key:
        raise RuntimeError(f"API key for {provider} not set")
    return api_key

class LLMSession:
    def __init__(self):
        self._history = []
        self._cache = None
        self._client = None
        self._model = None

    def reset(self):
        """Call this between prompts to clear history and delete the cache"""
        if self._cache and self._client:
            try:
                self._client.caches.delete(name=self._cache.name)
            except Exception:
                pass
        self._history = []
        self._cache = None

    def send_query(self, provider, model_name, prompt):
        api_key = get_api_key(provider)
        if provider == "gemini":
            return self._send_gemini(model_name, prompt, api_key)
        elif provider == "openai":
            return self._send_openai(model_name, prompt, api_key)
        elif provider == "anthropic":
            return self._send_anthropic(model_name, prompt, api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _send_gemini(self, model_name, prompt, api_key):
        """ Send query to gemini, including possible caching"""
        full_model = f"gemini-{model_name}"

        if self._client is None:
            self._client = genai.Client(api_key=api_key)
        self._model = full_model

        self._history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        # Send message, either with cache or general history
        try:
            if self._cache:
                response = self._client.models.generate_content(model=full_model, contents=[self._history[-1]], config=types.GenerateContentConfig(cached_content=self._cache.name))
            else:
                response = self._client.models.generate_content(model=full_model, contents=self._history)
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

        output = response.text
        self._history.append(types.Content(role="model", parts=[types.Part(text=output)]))

        # Try to create/refresh cache
        try:
            if self._cache:
                self._client.caches.delete(name=self._cache.name)
            self._cache = self._client.caches.create(model=full_model, config=types.CreateCachedContentConfig(contents=self._history, ttl="300s"))
        except Exception:
            # Content too short to cache
            self._cache = None

        return output

    def _send_openai(self, model_name, prompt, api_key):
        """ Send query to chatgpt. TODO no caching yet"""
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

    def _send_anthropic(self, model_name, prompt, api_key):
        """ Send query to claude. TODO no caching yet"""
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(model=model_name, messages=[{"role": "user", "content": prompt}])
        return "".join(block.text for block in response.content if block.type == "text")