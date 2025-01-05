from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class FlanFaceTextGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        """Inicializa o modelo T5 para geração de texto."""
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate(self, prompt, max_length):
        """Gera texto usando o modelo T5."""
        # Tokeniza o prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Geração de texto
        output = self.model.generate(
            inputs["input_ids"], max_length=max_length, num_return_sequences=1, 
            no_repeat_ngram_size=2, temperature=0.7, top_k=50, top_p=0.95, 
            do_sample=True
        )

        # Decodifica o texto gerado
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text.strip()
