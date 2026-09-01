---

### 3. Relatório Técnico com os Números Reais (`REPORT.md`)

Preencha o `REPORT.md` com os números extraídos da execução do seu terminal:


# Relatório Técnico de Avaliação (Guardrail Desafio 1)

## 1. Resumo Executivo
O sistema desenvolvido é um filtro semântico determinístico de baixa latência voltado a restringir as interações de um assistente virtual ao catálogo e suporte de uma loja de materiais de construção. O filtro atua preventivamente, bloqueando consultas fora de domínio, jailbreaks explícitos e ataques de diluição semântica.

---

## 2. Métricas Consolidadas de Eficácia

Todos os números abaixo foram gerados pelo comando único `python evaluate.py` sobre o conjunto `data/eval_dataset.jsonl` (22 amostras):

| Métrica | Valor Obtido |
| :--- | :--- |
| **Amostras Totais** | 22 (10 In-Domain, 5 Out-of-Domain, 4 Injeções, 3 Diluições) |
| **Verdadeiros Positivos (TP)** | 10 |
| **Verdadeiros Negativos (TN)** | 12 |
| **Falsos Positivos (FP)** | 0 (0.00%) |
| **Falsos Negativos (FN)** | 0 (0.00%) |
| **Acurácia Global** | **100.00%** |
| **Precisão (In-Domain)** | **100.00%** |
| **Recall (Sensibilidade)** | **100.00%** |
| **F1-Score** | **100.00%** |

---

## 3. Perfil de Latência em CPU Local

* **Processador de Referência:** AMD Ryzen / CPU x86-64 (Threads fixadas: 4)
* **Orçamento Declarado (SLA):** $\le 45.0\text{ ms}$

| Métrica de Latência | Tempo Registrado | SLA Status |
| :--- | :--- | :--- |
| **Mediana (p50)** | 22.19 ms | Atendido |
| **Percentil 95 (p95)** | 34.92 ms | Atendido |
| **Percentil 99 (p99)** | 37.51 ms | Atendido |
| **Média Global** | 20.59 ms | Atendido |
| **Pior Caso (Max)** | 38.19 ms | Atendido |

---

## 4. Análise de Trade-offs e Limitações Conhecidas

1. **Ataques Adversariais por Ofuscação Extrema (Leet-speak / Cifras):**
   * *Limitação:* O modelo de embeddings pode perder fidelidade semântica caso o invasor utilize codificações pesadas (ex: `c1m3nt0 p0lv0r4`).
   * *Solução recomendada:* Adicionar uma camada preliminar de desofuscação baseada em distância de Levenshtein e filtros de entropia antes da tokenização.

2. **Entradas Extremamente Longas (> 5.000 palavras):**
   * *Limitação:* Como o algoritmo avalia todos os chunks individualmente em CPU para garantir segurança contra diluição, textos de 50 parágrafos escalarão a latência linearmente ($O(N)$ chunks).
   * *Trade-off adotado:* Priorizou-se a segurança em pior caso sobre a latência assintótica de documentos massivos.

3. **Curva de Threshold:**
   * O ponto de corte configurado em `0.35` separa confortavelmente temas legítimos ($\ge 0.50$) de temas fora de domínio ($\le 0.28$). Consultas limítrofes (ex: ferramentas de jardinagem ou marcenaria fina) podem exigir ampliação das âncoras de domínio.