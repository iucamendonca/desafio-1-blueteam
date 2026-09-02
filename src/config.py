from pydantic import BaseModel, Field, ConfigDict


class GuardrailConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    
    encoder_model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Modelo multilíngue leve otimizado para semântica em PT-BR."
    )
    onnx_threads: int = Field(
        default=4,
        description="Número de threads na CPU para execução determinística."
    )
    
    
    latency_budget_ms: float = Field(
        default=45.0,
        description="Orçamento máximo declarado para CPU (cobre textos curtos e múltiplos parágrafos)."
    )

    
    domain_similarity_threshold: float = Field(
        default=0.35,
        description="Similaridade mínima de cosseno necessária com a melhor âncora de domínio."
    )

    
    max_chunk_words: int = Field(
        default=32,
        description="Tamanho máximo de palavras por sentença/chunk."
    )
    chunk_overlap_words: int = Field(
        default=8,
        description="Sobreposição entre janelas para sentenças longas."
    )
