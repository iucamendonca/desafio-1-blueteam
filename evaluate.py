"""
Script de avaliação determinística e medição de latência do Guardrail.
Executa o benchmark sobre o dataset e gera métricas estatísticas consolidadas.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from src.config import GuardrailConfig
from src.guardrail import DeterministicGuardrail


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dataset não encontrado: {file_path}")
    
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_benchmark(dataset_path: str = "data/eval_dataset.jsonl"):
    print("=" * 65)
    print(" INICIANDO BENCHMARK DO GUARDRAIL (EXECUÇÃO DETERMINÍSTICA)")
    print("=" * 65)

    config = GuardrailConfig()
    guardrail = DeterministicGuardrail(config)
    dataset = load_dataset(dataset_path)

    # Warm-up de 1 ciclo para que o carregamento inicial não distorça a latência
    guardrail.evaluate("Aquecimento de inferência.")

    latencies_ms = []
    results = []

    for idx, item in enumerate(dataset):
        res = guardrail.evaluate(item["text"])
        latencies_ms.append(res["latency_ms"])
        
        is_correct = (res["verdict"] == item["expected_verdict"])
        results.append({
            "id": idx + 1,
            "text": item["text"],
            "category": item["category"],
            "expected": item["expected_verdict"],
            "predicted": res["verdict"],
            "score": res["score"],
            "reason": res["reason"],
            "is_correct": is_correct,
            "latency_ms": res["latency_ms"]
        })

    # Classe Positiva: ALLOW (In-Domain Legítimo)
    # Classe Negativa: REJECT (Fora de domínio / Injeção / Diluição)
    tp = sum(1 for r in results if r["expected"] == "ALLOW" and r["predicted"] == "ALLOW")
    tn = sum(1 for r in results if r["expected"] == "REJECT" and r["predicted"] == "REJECT")
    fp = sum(1 for r in results if r["expected"] == "REJECT" and r["predicted"] == "ALLOW")
    fn = sum(1 for r in results if r["expected"] == "ALLOW" and r["predicted"] == "REJECT")

    total = len(results)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    latencies_arr = np.array(latencies_ms)
    p50 = np.percentile(latencies_arr, 50)
    p95 = np.percentile(latencies_arr, 95)
    p99 = np.percentile(latencies_arr, 99)
    mean_lat = np.mean(latencies_arr)
    max_lat = np.max(latencies_arr)

    print("\n--- MATRIZ DE CONFUSÃO & EFICÁCIA ---")
    print(f"Total de Amostras Avaliadas : {total}")
    print(f"Verdadeiros Positivos (TP)   : {tp} (Permitiu consultas legítimas)")
    print(f"Verdadeiros Negativos (TN)   : {tn} (Bloqueou ataques e temas fora)")
    print(f"Falsos Positivos (FP)        : {fp} (Deixou passar o que devia barrar)")
    print(f"Falsos Negativos (FN)        : {fn} (Barrou indevidamente cliente legítimo)")
    print("-" * 65)
    print(f"Acurácia Global              : {accuracy * 100:.2f}%")
    print(f"Precisão (In-Domain)         : {precision * 100:.2f}%")
    print(f"Recall (Sensibilidade)       : {recall * 100:.2f}%")
    print(f"F1-Score                     : {f1_score * 100:.2f}%")
    print("-" * 65)

    failures = [r for r in results if not r["is_correct"]]
    if failures:
        print("\n--- DETALHAMENTO DE FALHAS REGISTRADAS ---")
        for f in failures:
            print(f"ID #{f['id']} | Categoria: {f['category']}")
            print(f"  Texto     : {f['text']}")
            print(f"  Esperado  : {f['expected']} | Predito: {f['predicted']} (Score: {f['score']} | Threshold: {config.domain_similarity_threshold})")
            print(f"  Motivo    : {f['reason']}")
            print("-" * 65)

    print("\n--- PERFIL DE LATÊNCIA (CPU LOCAL) ---")
    print(f"Orçamento Declarado (SLA)   : <= {config.latency_budget_ms:.1f} ms")
    print(f"Latência Mediana (p50)       : {p50:.2f} ms")
    print(f"Latência Percentil 95 (p95)  : {p95:.2f} ms")
    print(f"Latência Percentil 99 (p99)  : {p99:.2f} ms")
    print(f"Latência Média               : {mean_lat:.2f} ms")
    print(f"Pior Caso Registrado (Max)   : {max_lat:.2f} ms")
    print("=" * 65)

    sla_status = "ATENDIDO COM SUCESSO" if p95 <= config.latency_budget_ms else "VIOLADO"
    print(f"Status do Orçamento de Latência: {sla_status}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação automatizada do Guardrail.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval_dataset.jsonl",
        help="Caminho para o arquivo .jsonl de avaliação."
    )
    args = parser.parse_args()
    run_benchmark(args.dataset)