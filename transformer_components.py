"""
Implementa as três sub-camadas do Encoder:
  - Scaled Dot-Product Attention
  - Conexão Residual + Layer Normalization
  - Feed-Forward Network (FFN)
"""

import numpy as np

# Hiperparâmetros
D_MODEL = 64          # dimensão do modelo  (paper: 512)
D_K     = 64          # dimensão de Q e K   (paper: 64 com 8 heads)
D_V     = 64          # dimensão de V
D_FF    = 256         # dimensão interna do FFN (paper: 2048; aqui 4×d_model)
EPSILON = 1e-6        # epsilon para LayerNorm


# Scaled Dot-Product Attention
class ScaledDotProductAttention:
    """
    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
    """

    def __init__(self, d_model: int, d_k: int, d_v: int):
        self.d_k = d_k
        # Matrizes de projeção inicializadas aleatoriamente
        self.W_Q = np.random.randn(d_model, d_k) * 0.01
        self.W_K = np.random.randn(d_model, d_k) * 0.01
        self.W_V = np.random.randn(d_model, d_v) * 0.01

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax numericamente estável ao longo do último eixo."""
        # Subtrai o máximo para estabilidade numérica
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x     = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Parâmetros -> X : np.ndarray  shape (batch, seq_len, d_model)

        Retorna -> output : np.ndarray  shape (batch, seq_len, d_v)
        """
        # Passo 2 – Projeções lineares
        Q = X @ self.W_Q          # (batch, seq, d_k)
        K = X @ self.W_K          # (batch, seq, d_k)
        V = X @ self.W_V          # (batch, seq, d_v)

        # Passo 3 – Produto escalar Q·K^T
        # K transposta nas duas últimas dimensões para broadcasting correto
        scores = Q @ K.transpose(0, 2, 1)   # (batch, seq, seq)

        # Passo 4 – Scaling
        scores = scores / np.sqrt(self.d_k)

        # Passo 5 – Softmax (implementação própria)
        weights = self._softmax(scores)      # (batch, seq, seq)

        # Passo 6 – Produto pelos valores
        output = weights @ V                 # (batch, seq, d_v)
        return output


# Conexão residual + Layer Normalization

class LayerNorm:
    """
    LayerNorm normaliza na dimensão dos features (último eixo).
    Parâmetros treináveis gamma (escala) e beta (deslocamento)
    são inicializados como 1 e 0 respectivamente.
    """

    def __init__(self, d_model: int, epsilon: float = EPSILON):
        self.epsilon = epsilon
        self.gamma   = np.ones(d_model)    # escala  (treinável)
        self.beta    = np.zeros(d_model)   # bias    (treinável)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parâmetros
        ----------
        x : np.ndarray  shape (batch, seq_len, d_model)

        Retorna
        -------
        x_norm : np.ndarray  shape (batch, seq_len, d_model)
        """
        mean = np.mean(x, axis=-1, keepdims=True)          # (batch, seq, 1)
        var  = np.var (x, axis=-1, keepdims=True)          # (batch, seq, 1)
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)  # normalização
        return self.gamma * x_norm + self.beta             # escala e bias


def residual_add_norm(x: np.ndarray,
                      sublayer_output: np.ndarray,
                      layer_norm: LayerNorm) -> np.ndarray:
    """
    Implementa:  LayerNorm( x + Sublayer(x) )
    """
    return layer_norm.forward(x + sublayer_output)


# Feed-Forward Network

class FeedForwardNetwork:
    """
    FFN(x) = max(0, x W1 + b1) W2 + b2

    Expande d_model → d_ff → d_model.
    """

    def __init__(self, d_model: int, d_ff: int):
        # Pesos e biases das duas transformações lineares
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU: max(0, x)"""
        return np.maximum(0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parâmetros
        x : np.ndarray  shape (batch, seq_len, d_model)

        Retorna
        output : np.ndarray  shape (batch, seq_len, d_model)
        """
        hidden = self._relu(x @ self.W1 + self.b1)   # (batch, seq, d_ff)

        output = hidden @ self.W2 + self.b2           # (batch, seq, d_model)
        return output


# Smoke Test – valida shapes de cada componente isoladamente
if __name__ == "__main__":
    np.random.seed(42)

    BATCH, SEQ = 1, 5
    X_test = np.random.randn(BATCH, SEQ, D_MODEL)
    print(f"Tensor X_test shape : {X_test.shape}\n")

    # Attention
    attention = ScaledDotProductAttention(D_MODEL, D_K, D_V)
    att_out   = attention.forward(X_test)
    print(f"[Attention]  saída shape : {att_out.shape}  "
          f"(esperado: ({BATCH}, {SEQ}, {D_V}))")

    # Add & Norm pós-atenção
    ln1    = LayerNorm(D_MODEL)
    norm1  = residual_add_norm(X_test, att_out, ln1)
    print(f"[Add+Norm 1] saída shape : {norm1.shape}  "
          f"(esperado: ({BATCH}, {SEQ}, {D_MODEL}))")

    # FFN
    ffn     = FeedForwardNetwork(D_MODEL, D_FF)
    ffn_out = ffn.forward(norm1)
    print(f"[FFN]        saída shape : {ffn_out.shape}  "
          f"(esperado: ({BATCH}, {SEQ}, {D_MODEL}))")

    # Add & Norm pós-FFN
    ln2   = LayerNorm(D_MODEL)
    norm2 = residual_add_norm(norm1, ffn_out, ln2)
    print(f"[Add+Norm 2] saída shape : {norm2.shape}  "
          f"(esperado: ({BATCH}, {SEQ}, {D_MODEL}))")

    print("\n[Passo 2 concluído ✓]")
