from src.normalizer import TextNormalizer


def test_clean_text_removes_invisible_and_extra_spaces():
    normalizer = TextNormalizer()
    raw_input = "  Qual   a melhor\nargamassa\tpara piso? \x00 "
    expected = "Qual a melhor argamassa para piso?"
    assert normalizer.clean_text(raw_input) == expected


def test_short_input_returns_single_chunk():
    normalizer = TextNormalizer(max_words=10, overlap_words=2)
    text = "Cimento CP II para laje residencial."
    chunks = normalizer.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_input_sliding_window():
    normalizer = TextNormalizer(max_words=6, overlap_words=2)
    # 10 palavras -> step = 4 -> Chunks: [0:6] (6 palavras), [4:10] (6 palavras)
    text = "um dois tres quatro cinco seis sete oito nove dez"
    chunks = normalizer.chunk_text(text)
    
    assert len(chunks) == 2
    assert chunks[0] == "um dois tres quatro cinco seis"
    assert chunks[1] == "cinco seis sete oito nove dez"