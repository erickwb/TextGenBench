import json
import os

def save_results(results_df, output_dir):
    """Salva os resultados em CSV."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Resultados salvos em {csv_path}")

def load_prompt(json_file):
    """Função para carregar o prompt e as instruções do arquivo JSON"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['prompt'], data['instrucoes']

# Função para carregar prompt, instruções e referência do JSON
def load_prompt_and_reference(file_path):
    import json

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = data.get("prompt")
    instrucoes = data.get("instrucoes", [])
    referencia = data.get("referencia")

    return prompt, instrucoes, referencia