> **TODO:** Expand this guide with detailed content.

# NLP Fundamentals

Natural Language Processing concepts from classical methods to pre-transformer deep learning — the foundation for understanding modern LLMs.

## Topics to Cover

### Text Preprocessing
- **Tokenization** — word-level, subword (BPE, WordPiece), character-level
- **Stemming & Lemmatization** — Porter stemmer, WordNet lemmatizer
- **Stop words, lowercasing, punctuation removal** — when helpful vs harmful
- **Text normalization** — unicode handling, spell correction

### Text Representation
- **Bag of Words (BoW)** — term frequency, TF-IDF weighting
- **Word embeddings** — Word2Vec (CBOW, Skip-gram), GloVe, FastText
- **Contextual embeddings** — ELMo, BERT (bidirectional), GPT (unidirectional)
- **Sentence/document embeddings** — mean pooling, [CLS] token, Sentence-BERT

### Classical NLP Tasks
- **Text classification** — sentiment analysis, spam detection, topic classification
- **Named Entity Recognition (NER)** — BIO tagging, sequence labeling
- **Part-of-speech tagging** — sequence labeling approaches
- **Machine translation** — encoder-decoder, attention mechanism origin story
- **Summarization** — extractive vs abstractive
- **Question answering** — extractive (span extraction) vs generative

### Language Modeling
- **N-gram models** — bigram, trigram, smoothing techniques
- **Neural language models** — RNN-based, perplexity as evaluation metric
- **Masked language modeling** — BERT's pre-training objective (MLM + NSP)
- **Causal language modeling** — GPT's pre-training objective (next token prediction)
- **Pre-train → Fine-tune paradigm** — BERT/GPT revolution

### Key Models Timeline
- Word2Vec (2013) → GloVe (2014) → ELMo (2018) → BERT (2018) → GPT-2 (2019) → T5 (2019) → GPT-3 (2020) → ChatGPT (2022) → GPT-4 (2023)

### Interview Questions
- Word2Vec vs GloVe — differences and when to use each?
- How does TF-IDF work and what are its limitations?
- BERT vs GPT — architectural differences and use cases?
- What is the difference between extractive and abstractive summarization?
- Explain attention mechanism in the context of machine translation
