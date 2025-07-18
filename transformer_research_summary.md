**Title: Deep Dive into the Transformer: Detailed Insights from 'Attention Is All You Need'**

**1. Introduction**
This document is an in-depth summary of the Transformer architecture as introduced in the landmark paper *"Attention Is All You Need"* by Vaswani et al. (2017). The Transformer model replaced recurrent and convolutional architectures in sequence modeling tasks by leveraging attention mechanisms exclusively. This write-up captures both theoretical understanding and practical implementation insights.

---

**2. Key Components of the Transformer**

**2.1 Tokenization and Vocabulary**
- Tokenization is performed using subword methods like Byte-Pair Encoding (BPE) or WordPiece.
- The tokenizer builds a fixed vocabulary before training.
- Source (`src_vocab_size`) and target (`tgt_vocab_size`) vocabularies may be identical or separate, depending on whether weight sharing or joint training is used.

**2.2 Embeddings**
- Input tokens are converted to vectors using an embedding matrix of shape `[vocab_size, d_model]`.
- Embeddings are scaled by \( \sqrt{d_{model}} \) to stabilize gradients.
- Positional encodings are added to inject sequence order information.
- During training, the embedding matrix is learned and often shared with the output projection layer.

**2.3 Positional Encoding**
- Since the model lacks recurrence, positional encodings are added to embeddings.
- These are fixed sinusoidal functions:
  \[ PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{model}}) \]
  \[ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{model}}) \]
- The result is a `[seq_len, d_model]` matrix that is added to token embeddings.

**2.4 Scaled Dot-Product Attention**
- Given queries Q, keys K, and values V, attention is computed as:
  \[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]
- Shapes:
  - Q, K, V: `[batch_size, num_heads, seq_len, d_k]`
  - Attention matrix: `[batch_size, num_heads, seq_len_q, seq_len_k]`
- An optional attention mask is added before the softmax to mask out invalid connections (e.g., future tokens).

**2.5 Multi-Head Attention**
- Projects Q, K, V from `[batch_size, seq_len, d_model]` to `[batch_size, num_heads, seq_len, d_k]`.
- Each head computes scaled dot-product attention in parallel.
- Outputs from all heads are concatenated and linearly projected back to `[batch_size, seq_len, d_model]`.
- This allows the model to jointly attend to information from different representation subspaces.

**2.6 Feed-Forward Networks (FFN)**
- Each position independently passes through a two-layer feed-forward network:
  \[ FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2 \]
- Shapes:
  - Input: `[batch_size, seq_len, d_model]`
  - Hidden: `[batch_size, seq_len, d_ff]`
  - Output: `[batch_size, seq_len, d_model]`

**2.7 Encoder and Decoder Structure**
- **Encoder Stack (×6)**:
  - Each layer contains:
    - Multi-head self-attention
    - Feed-forward network
    - Residual connections + LayerNorm
- **Decoder Stack (×6)**:
  - Each layer contains:
    - Masked multi-head self-attention (prevents seeing future tokens)
    - Encoder-decoder cross attention (Q from decoder, K/V from encoder output)
    - Feed-forward network
    - Residual connections + LayerNorm
- Only the final encoder output is passed to **every** decoder layer.

**2.8 Output Projection and Softmax**
- The decoder output has shape `[batch_size, seq_len, d_model]`
- It is projected using the transposed embedding matrix (`[d_model, vocab_size]`) to get logits
- A softmax is applied to produce probabilities over the vocabulary for each token position.

---

**3. Regularization Techniques**
- **Dropout**:
  - Applied after each sublayer (attention and FFN)
  - Also applied to summed input + positional encodings
  - Typical dropout rate: 0.1
- **Label Smoothing**:
  - Replaces one-hot targets with a smoothed version to prevent overconfidence
  - Example: for smoothing \( \epsilon_{ls} = 0.1 \), target label is 0.9 for the correct class, 0.1 distributed over others

---

**4. Implementation Insights**

**4.1 Efficient Multi-Head Attention**
- Q, K, V are computed via shared `[d_model, d_model]` linear layers, not per-head layers.
- After projection, tensors are reshaped to `[batch, num_heads, seq_len, d_k]`.
- Attention is computed in parallel for all heads and concatenated afterward.

**4.2 Decoder Masking**
- During training, an upper-triangular mask (with `-inf`) is applied to `att_matrix` before softmax to ensure auto-regression.

**4.3 Encoder-Decoder Interaction**
- Encoder output is reused in all decoder layers at the cross-attention block.
- This avoids recomputing attention across layers.

**4.4 Dimension Matching**
- FFN: `[d_model → d_ff → d_model]`
- Attention projections: `[d_model → d_model]`, split into heads: `[d_model → num_heads × d_k]`
- Output projection: `[d_model → d_model]`
- Final logits: `[d_model → vocab_size]`

---

**5. Architectural Summary**
| Layer | Input Shape | Output Shape | Notes |
|-------|-------------|--------------|-------|
| Embedding + PE | [B, L] → [B, L, d_model] | Scaled and positionally encoded |
| Multi-Head Attention | [B, L, d_model] → [B, L, d_model] | Includes dropout and residual |
| Feed-Forward | [B, L, d_model] → [B, L, d_model] | Applied per position |
| Encoder Stack | [B, L, d_model] → [B, L, d_model] | Final encoder output reused by decoder |
| Decoder Stack | [B, L, d_model] → [B, L, d_model] | Uses encoder output in cross-attention |
| Output Projection | [B, L, d_model] → [B, L, vocab_size] | Before softmax |

---

**6. Conclusion**
Studying the Transformer architecture in detail has given me a grounded understanding of its core mechanisms:
- Multi-head attention enables parallel attention across multiple subspaces.
- Feed-forward layers increase capacity without breaking parallelism.
- Positional encodings and residual connections stabilize training.
- Implementation-wise, the architecture is modular and efficiently parallelizable.
