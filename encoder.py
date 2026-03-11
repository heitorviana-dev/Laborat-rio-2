"""
Integra os Passos 1 e 2 e empilha N=6 camadas do Encoder.
Fluxo por camada:
    1. X_att   = SelfAttention(X)
    2. X_norm1 = LayerNorm(X + X_att)
    3. X_ffn   = FFN(X_norm1)
    4. X_out   = LayerNorm(X_norm1 + X_ffn)
    5. X       = X_out
"""

import numpy as np
import pandas as pd


from transformer_components import (
    ScaledDotProductAttention,
    FeedForwardNetwork,
    LayerNorm,
    residual_add_norm,
    D_MODEL, D_K, D_V, D_FF,
)

np.random.seed(42)


# ENCODER LAYER  (uma única camada)

class EncoderLayer:
    """
    Uma camada do Encoder conforme 'Attention Is All You Need'.
    Contém:
      - Self-Attention  +  Add & Norm
      - Feed-Forward    +  Add & Norm
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, d_ff: int):
        self.attention = ScaledDotProductAttention(d_model, d_k, d_v)
        self.ffn       = FeedForwardNetwork(d_model, d_ff)
        self.ln1       = LayerNorm(d_model)
        self.ln2       = LayerNorm(d_model)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Parâmetros
        X : np.ndarray  shape (batch, seq_len, d_model)

        Retorna
        X_out : np.ndarray  shape (batch, seq_len, d_model)
        """
        # Passo 1 – Self-Attention
        X_att   = self.attention.forward(X)

        # Passo 2 – Add & Norm pós-atenção
        X_norm1 = residual_add_norm(X, X_att, self.ln1)

        # Passo 3 – Feed-Forward
        X_ffn   = self.ffn.forward(X_norm1)

        # Passo 4 – Add & Norm pós-FFN
        X_out   = residual_add_norm(X_norm1, X_ffn, self.ln2)

        return X_out   # Passo 5: vira input da próxima camada


# Transformer Encoder  (pilha de N camadas)

class TransformerEncoder:
    """
    Pilha de N camadas idênticas do Encoder.
    Cada camada possui seus próprios pesos — não são compartilhados.
    """

    def __init__(self, n_layers: int, d_model: int,
                 d_k: int, d_v: int, d_ff: int):
        self.layers = [
            EncoderLayer(d_model, d_k, d_v, d_ff)
            for _ in range(n_layers)
        ]

    def forward(self, X: np.ndarray,
                verbose: bool = True) -> np.ndarray:
        """
        Passa o tensor X por todas as N camadas sequencialmente.

        Parâmetros
        X       : np.ndarray  shape (batch, seq_len, d_model)
        verbose : bool        imprime shape em cada camada se True

        Retorna
        Z : np.ndarray  shape (batch, seq_len, d_model)
        """
        Z = X
        for i, layer in enumerate(self.layers, start=1):
            Z = layer.forward(Z)
            if verbose:
                print(f"  Camada {i}: shape de saída = {Z.shape}")
        return Z

# Pipeline Completo  (Passo 1 + Passo 2 + Passo 3)

def main():
    # Passo 1: Preparação dos Dados
    print("=" * 55)
    print("PASSO 1 – Preparação dos Dados")
    print("=" * 55)

    vocabulario = {
        "o": 0, "banco": 1, "bloqueou": 2, "cartao": 3,
        "cliente": 4, "pagou": 5, "a": 6, "fatura": 7,
    }

    df_vocab = pd.DataFrame(
        list(vocabulario.items()), columns=["palavra", "id"]
    )
    print(df_vocab.to_string(index=False))

    frase   = ["o", "cliente", "pagou", "a", "fatura"]
    ids     = [vocabulario[p] for p in frase]
    print(f"\nFrase : {frase}")
    print(f"IDs   : {ids}")

    VOCAB_SIZE = len(vocabulario)
    embedding_table = np.random.randn(VOCAB_SIZE, D_MODEL)

    BATCH_SIZE = 1
    SEQ_LEN    = len(ids)

    X = embedding_table[ids][np.newaxis, :, :]
    print(f"\nTensor X shape (entrada): {X.shape}")

    # Passo 3: Encoder com 6 Camadas
    print("\n" + "=" * 55)
    print("PASSO 3 – Encoder Stack (N=6 camadas)")
    print("=" * 55)

    N_LAYERS = 6
    encoder  = TransformerEncoder(N_LAYERS, D_MODEL, D_K, D_V, D_FF)

    Z = encoder.forward(X, verbose=True)

    # Validação de Sanidade
    print("\n" + "=" * 55)
    print("VALIDAÇÃO DE SANIDADE")
    print("=" * 55)
    shape_ok = Z.shape == X.shape
    print(f"  Shape de entrada  : {X.shape}")
    print(f"  Shape de saída  Z : {Z.shape}")
    print(f"  Shapes preservados: {'✓ SIM' if shape_ok else '✗ NÃO'}")

    # Verificação numérica: Z deve diferir de X (pesos contextualizados)
    diferenca = np.mean(np.abs(Z - X))
    print(f"  Diferença média |Z - X| : {diferenca:.6f}  "
          f"(>0 confirma contextualizacao)")

    print(f"\nRepresentação Z (primeiros 8 valores do token 0):")
    print(f"  {Z[0, 0, :8]}")
    print("\n[Passo 3 concluído ✓  —  Encoder Forward Pass completo!]")

    return Z

if __name__ == "__main__":
    Z = main()
