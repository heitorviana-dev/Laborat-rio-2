# Transformer Encoder — From Scratch

**Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
**Professor:** Prof. Dimmy Magalhães  
**Instituição:** iCEV - Instituto de Ensino Superior  

Implementação do Forward Pass de um Transformer Encoder completo, baseado no artigo  
*"Attention Is All You Need"* (Vaswani et al., 2017), usando apenas `Python 3`, `numpy` e `pandas`.

---

## Estrutura do Projeto

```
transformer-encoder/
├── data_preparation.py              # Vocabulário, embeddings e tensor X
├── transformer_components.py   # Attention, LayerNorm e FFN
├── encoder.py   # Pipeline completo com N=6 camadas
└── README.md
```

---

## Como Rodar

### Pré-requisitos

```bash
pip install numpy pandas
```

### Executar cada passo individualmente

```bash
# Passo 1 — Preparação dos dados
python data_preparation.py

# Passo 2 — Smoke test dos componentes matemáticos
python transformer_components.py

# Passo 3 — Encoder completo (integra tudo)
python encoder.py
```

O arquivo principal é `encoder.py`. Ele importa os  
componentes de `transformer_components.py` e executa o pipeline de ponta a ponta.

---

## Arquitetura Implementada

```
Entrada (frase)
     │
     ▼
Embedding Table  →  X : (Batch=1, SeqLen=5, d_model=64)
     │
     ▼
┌─────────────────────────────────────┐
│  EncoderLayer  ×  6                 │
│                                     │
│  X_att   = SelfAttention(X)         │
│  X_norm1 = LayerNorm(X + X_att)     │
│  X_ffn   = FFN(X_norm1)             │
│  X_out   = LayerNorm(X_norm1+X_ffn) │
└─────────────────────────────────────┘
     │
     ▼
  Z : (Batch=1, SeqLen=5, d_model=64)
  (representação contextualizada)
```

### Hiperparâmetros

| Parâmetro | Este lab | Paper original |
|-----------|----------|----------------|
| `d_model` | 64       | 512            |
| `d_k / d_v` | 64     | 64             |
| `d_ff`    | 256      | 2048           |
| `N layers`| 6        | 6              |

---

## Componentes Matemáticos

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax( Q K^T / √d_k ) V
```
- Projeta X em Q, K, V via matrizes W_Q, W_K, W_V
- Softmax implementada manualmente com `np.exp` (estabilidade numérica via shift pelo máximo)

### Layer Normalization
```
LayerNorm(x) = γ · (x − μ) / √(σ² + ε) + β
```
- Opera no último eixo (features), diferente do Batch Norm
- `ε = 1e-6` para evitar divisão por zero

### Feed-Forward Network
```
FFN(x) = max(0, x W₁ + b₁) W₂ + b₂
```
- Expansão `d_model → d_ff`, ativação ReLU, contração `d_ff → d_model`

---

## Validação de Sanidade

O tensor mantém o shape `(Batch, SeqLen, d_model)` ao longo de todas as 6 camadas,  
confirmando que a arquitetura está correta. A diferença média `|Z − X| > 0` confirma  
que os pesos foram aplicados e os vetores foram contextualizados.

---

## Nota de Integridade Acadêmica

Claude (Anthropic) foi consultado como ferramenta auxiliar para revisão de  
estrutura de código e sintaxe NumPy, conforme permitido pelo Contrato Pedagógico.  
A lógica, implementação matemática e decisões de arquitetura são de autoria do aluno.
