import time
import os
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

def calculate_metrics(generator, prompt, max_length, reference, model_name):
    """Calcula métricas como tempo de geração, BLEU, ROUGE e similaridade."""

    # Medir o tempo de geração
    start_time = time.time()
    generated_text = generator.generate(prompt, max_length)
    elapsed_time = time.time() - start_time

    # Tokenização consistente
    tokenized_reference = word_tokenize(reference)
    tokenized_generated = word_tokenize(generated_text)

    # BLEU Score
    bleu_score = sentence_bleu([tokenized_reference], tokenized_generated, weights=(0.6, 0.3, 0.1, 0))

    # ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, generated_text)
    rouge1 = rouge_scores['rouge1'].fmeasure
    rouge2 = rouge_scores['rouge2'].fmeasure
    rougeL = rouge_scores['rougeL'].fmeasure

    # Similaridade Semântica
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings_reference = model.encode(reference, convert_to_tensor=True)
    embeddings_generated = model.encode(generated_text, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings_reference, embeddings_generated).item()

    return {
        "Model": model_name,
      "Prompt": str(prompt)[:100] + ("..." if len(str(prompt)) > 100 else ""),
        "Generated Text": generated_text,
        "Time (s)": elapsed_time,
        "BLEU": bleu_score,
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL,
        "Semantic Similarity": similarity_score
    }


def append_results(new_results_df, output_dir):
    """Adiciona novos resultados ao arquivo existente, mantendo os anteriores."""
    os.makedirs(output_dir, exist_ok=True)
    results_file = f"{output_dir}/benchmark_results.csv"

    if os.path.exists(results_file):
        # Carrega os resultados existentes e concatena com os novos
        existing_results = pd.read_csv(results_file)
        combined_results = pd.concat([existing_results, new_results_df], ignore_index=True)
    else:
        # Se o arquivo não existir, apenas salva os novos resultados
        combined_results = new_results_df

    # Salva os resultados atualizados
    combined_results.to_csv(results_file, index=False)
    print(f"Resultados atualizados salvos em {results_file}")

