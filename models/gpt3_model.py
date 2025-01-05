import openai

class GPT3TextGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate(self, prompt, max_length):
        """Gera texto usando GPT-3.5 ou GPT-4 via API OpenAI."""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_length
        )
        return response["choices"][0]["message"]["content"].strip()