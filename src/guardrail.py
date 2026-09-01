"""
Motor de inferência e decisão do Guardrail com Classificação Multi-Âncora.
Execução local determinística em CPU.
"""

import time
from typing import List, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config import GuardrailConfig
from src.normalizer import TextNormalizer


DOMAIN_ANCHORS = [
    "Qual cimento, argamassa, rejunte ou concreto usar para assentamento de piso, laje e contrapiso?",
    "Cálculo de quantidade e rendimento de sacos de cimento, areia e brita para obras.",
    "Tubos, conexões de PVC, ralos e registros para encanamento, esgoto e hidráulica predial.",
    "Fiação elétrica, cabos flexíveis, disjuntores, conduítes e tomadas residenciais.",
    "Impermeabilização de lajes, baldrames, umidade, infiltração e manta asfáltica.",
    "Ferramentas elétricas manuais, furadeira, martelete, serra mármore, lixadeira, disco diamantado, brocas e marcas Bosch, Makita e DeWalt.",
    "Preços, catálogo, orçamento de materiais de construção, ferramentas, valores e disponibilidade de estoque.",
    "Tintas acrílicas, látex, massa corrida, primer, pincéis, rolos e verniz para pintura.",
    "Tijolos, blocos de concreto, ferro e materiais básicos de alvenaria e fundação.",
    "Prazos de entrega de materiais de construção, cálculo de frete e cotação de caçambas.",
    "Tipos de parafusos, buchas, pregos, porcas e fixação para alvenaria e drywall.",
    "Portas, janelas, fechaduras, telhas de cerâmica, fibrocimento e estruturas para telhado."
]

EXPLICIT_ATTACK_TRIGGERS = [
    "ignore previous instructions",
    "ignore all instructions",
    "system prompt",
    "dan mode",
    "jailbreak",
    "modo desenvolvedor",
    "desconsidere as regras anteriores",
    "aja como um médico",
    "aja como um hacker"
]


class DeterministicGuardrail:
    def __init__(self, config: GuardrailConfig = None):
        self.config = config or GuardrailConfig()
        self.normalizer = TextNormalizer(
            max_words=self.config.max_chunk_words,
            overlap_words=self.config.chunk_overlap_words
        )
        
        torch.manual_seed(42)
        torch.set_num_threads(self.config.onnx_threads)
        
        self.encoder = SentenceTransformer(self.config.encoder_model_name, device="cpu")
        self.anchor_embeddings = self._compute_anchor_embeddings()

    def _compute_anchor_embeddings(self) -> np.ndarray:
        """Gera e normaliza a matriz de embeddings para cada âncora de domínio."""
        embeddings = self.encoder.encode(
            DOMAIN_ANCHORS,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def _calculate_max_similarity(self, chunk_embedding: np.ndarray) -> float:
        """Calcula a maior similaridade de cosseno contra todas as âncoras."""
        # Produto escalar contra a matriz de âncoras: shape (N_anchors,)
        similarities = np.dot(self.anchor_embeddings, chunk_embedding)
        return float(np.max(similarities))

    def evaluate(self, text: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        cleaned_text = self.normalizer.clean_text(text)
        
        # 1. Borda: Texto Vazio
        if not cleaned_text:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "verdict": "REJECT",
                "score": 0.0,
                "reason": "EMPTY_INPUT",
                "latency_ms": round(elapsed_ms, 3)
            }

        # 2. Varredura Estrutural de Injeção
        lowered_text = cleaned_text.lower()
        for trigger in EXPLICIT_ATTACK_TRIGGERS:
            if trigger in lowered_text:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "verdict": "REJECT",
                    "score": 0.0,
                    "reason": "EXPLICIT_PROMPT_INJECTION",
                    "latency_ms": round(elapsed_ms, 3)
                }

        # 3. Divisão em Sentenças/Chunks
        chunks = self.normalizer.chunk_text(cleaned_text)
        chunk_embeddings = self.encoder.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        min_score = 1.0
        worst_chunk_idx = 0

        # 4. Avaliação por Pior Caso (Worst-Case Pooling)
        for idx, emb in enumerate(chunk_embeddings):
            score = self._calculate_max_similarity(emb)
            if score < min_score:
                min_score = score
                worst_chunk_idx = idx

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if min_score < self.config.domain_similarity_threshold:
            return {
                "verdict": "REJECT",
                "score": round(min_score, 4),
                "threshold": self.config.domain_similarity_threshold,
                "reason": "OUT_OF_DOMAIN_SEMANTIC",
                "failing_chunk_index": worst_chunk_idx,
                "latency_ms": round(elapsed_ms, 3)
            }

        return {
            "verdict": "ALLOW",
            "score": round(min_score, 4),
            "threshold": self.config.domain_similarity_threshold,
            "reason": "IN_DOMAIN_CONFIRMED",
            "latency_ms": round(elapsed_ms, 3)
        }

    def evaluate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.evaluate(t) for t in texts]