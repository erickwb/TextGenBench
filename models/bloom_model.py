from transformers import AutoModelForCausalLM, AutoTokenizer

class BLOOMTextGenerator:
    def __init__(self, model_name="bigscience/bloom-560m"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt, max_length):
        """Gera texto a partir de um prompt usando BLOOM."""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=max_length, do_sample=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

