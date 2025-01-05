import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Benchmark para Modelos de Geração de Texto")
    parser.add_argument("--model", type=str, required=True, help="Modelo a ser utilizado (ex: gpt2, gpt3, bloom, etc.)")
    parser.add_argument("--api_key", type=str, required=False, help="Chave de API para modelos como GPT-3 ou GPT-4")
    parser.add_argument("--length", type=int, default=100, help="Comprimento máximo do texto gerado")
    parser.add_argument("--repetitions", type=int, default="20", help="Numero de vezes que o modelo será executado")
    parser.add_argument("--prompt_file", type=str, required=True, help="Caminho para o arquivo JSON com o prompt e instruções")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Diretório para salvar os resultados")


    return parser.parse_args()
