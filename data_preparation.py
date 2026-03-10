"""
Laboratório 2 - Construindo o Transformer Encoder From Scratch
Disciplina: Tópicos em Inteligência Artificial
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Vocabulário simulado com pandas DataFrame
vocabulario = {
    "o": 0,
    "banco": 1,
    "bloqueou": 2,
    "cartao": 3,
    "cliente": 4,
    "pagou": 5,
    "a": 6,
    "fatura": 7,
}

df_vocab = pd.DataFrame(
    list(vocabulario.items()), columns=["palavra", "id"]
)
print("=== Vocabulário ===")
print(df_vocab.to_string(index=False))

# Frase de entrada → lista de IDs
frase = ["o", "cliente", "pagou", "a", "fatura"]
ids_frase = [vocabulario[palavra] for palavra in frase]
print(f"\nFrase     : {frase}")
print(f"IDs       : {ids_frase}")

# Tabela de Embeddings
VOCAB_SIZE = len(vocabulario)
D_MODEL = 64          # paper usa 512; porém usaremos 64 para CPU

embedding_table = np.random.randn(VOCAB_SIZE, D_MODEL)
print(f"\nTabela de Embeddings shape : {embedding_table.shape}  "
      f"(vocab_size={VOCAB_SIZE}, d_model={D_MODEL})")

# Tensor de entrada X  →  (Batch, SeqLen, d_model)
BATCH_SIZE = 1
SEQ_LEN = len(ids_frase)

# Busca os vetores das palavras da frase na tabela
X = embedding_table[ids_frase]          # (SeqLen, d_model)
X = X[np.newaxis, :, :]                  # (1, SeqLen, d_model)

print(f"\nTensor X shape : {X.shape}  "
      f"(batch={BATCH_SIZE}, seq_len={SEQ_LEN}, d_model={D_MODEL})")
print("\n[Passo 1 concluído ✓]")
