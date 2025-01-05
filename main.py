import os
import pandas as pd
from config import get_args
from utils import load_prompt_and_reference
from metrics import calculate_metrics, append_results
from visualization import plot_time_vs_execution, plot_bleu_vs_execution
from models import GPT2TextGenerator, GPT3TextGenerator, BLOOMTextGenerator, FlanFaceTextGenerator, GPT4TextGenerator

def get_model(model_type, api_key=None):
    """Retorna o gerador de texto apropriado baseado no modelo."""
    if model_type == "gpt2":
        return GPT2TextGenerator()
    elif model_type == "gpt3":
        if not api_key:
            raise ValueError("API key é necessária para GPT-3.")
        return GPT3TextGenerator(api_key)
    elif model_type == "bloom":
        return BLOOMTextGenerator()
 #python main.py --model llama --api_key "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" --length 200 --prompt_file prompt.txt --output_dir outputs/
    elif model_type == "flan-t5":
        return FlanFaceTextGenerator()
    elif model_type == "gpt4":
        return GPT4TextGenerator(api_key)
    else:
        raise ValueError(f"Modelo desconhecido: {model_type}")



def run_benchmark(generator, prompt, max_length, repetitions, reference, model_name, instructions):
    """Executa o benchmark coletando métricas para várias execuções."""
    # Combine as instruções ao prompt para enviá-las ao modelo
    full_prompt = f"{instructions}\n\n{prompt}"
    
    results = []
    for _ in range(repetitions):
        metrics = calculate_metrics(generator, full_prompt, max_length, reference, model_name)
        results.append(metrics)
    return pd.DataFrame(results)


def main():
    args = get_args()

    # Carrega o prompt
    prompt, instrucoes, referencia = load_prompt_and_reference(args.prompt_file)

    # Inicializa o gerador de texto
    generator = get_model(args.model, args.api_key)

    # Executa o benchmark
    print(f"Executando benchmark para o modelo {args.model}...")
    results_df = run_benchmark(
        generator, 
        prompt, 
        args.length, 
        repetitions=args.repetitions, 
        reference=referencia, 
        model_name=args.model, 
        instructions=" ".join(instrucoes)  # Concatena as instruções
    )

    # Salva os resultados
    append_results(results_df, args.output_dir)

    # Plota os gráficos
    plot_time_vs_execution(results_df, args.output_dir)
    plot_bleu_vs_execution(results_df, args.output_dir)

if __name__ == "__main__":
    main()
