import matplotlib.pyplot as plt

def plot_time_vs_execution(results_df, output_path):
    """Plota o tempo de execução por iteração."""
    plt.figure()
    plt.plot(results_df.index, results_df["Time (s)"], marker="o", label="Tempo (s)")
    plt.title("Tempo de Execução por Iteração")
    plt.xlabel("Iteração")
    plt.ylabel("Tempo (s)")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/time_execution.png")
    plt.show()

def plot_bleu_vs_execution(results_df, output_path):
    """Plota o BLEU Score por iteração."""
    plt.figure()
    plt.plot(results_df.index, results_df["BLEU"], marker="o", label="BLEU Score")
    plt.title("BLEU Score por Iteração")
    plt.xlabel("Iteração")
    plt.ylabel("BLEU Score")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/bleu_score.png")
    plt.show()
