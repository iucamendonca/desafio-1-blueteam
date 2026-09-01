"""
Normalização de texto e Janelamento Semântico (Chunking).
Garante determinismo absoluto e divisão uniforme de textos curtos e longos.
"""

import re
import unicodedata
from typing import List


class TextNormalizer:
    def __init__(self, max_words: int = 32, overlap_words: int = 8):
        self.max_words = max_words
        self.overlap_words = overlap_words

    def clean_text(self, text: str) -> str:
        """Limpa caracteres não-imprimíveis e normaliza espaços."""
        if not text:
            return ""
        
        normalized = unicodedata.normalize("NFC", text)
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", normalized)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def chunk_text(self, text: str) -> List[str]:
        """
        Divide o texto prioritariamente em sentenças completas.
        Se uma sentença for excessivamente longa, aplica a janela deslizante.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        # Divide por quebras de sentença preservando o sentido
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        # Se o texto era uma única linha curta sem pontuação final
        if not sentences:
            sentences = [cleaned]

        chunks = []
        for sentence in sentences:
            words = sentence.split(" ")
            if len(words) <= self.max_words:
                chunks.append(sentence)
            else:
                # Janela deslizante para sentenças longas
                step = self.max_words - self.overlap_words
                for i in range(0, len(words), step):
                    sub_chunk = " ".join(words[i:i + self.max_words])
                    if sub_chunk:
                        chunks.append(sub_chunk)
                    if i + self.max_words >= len(words):
                        break

        return chunks