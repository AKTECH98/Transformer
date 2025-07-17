import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class TransformerAttention(nn.Module):
    """
    Transformer Scaled Dot Product Attention Module
    Args:
        d_model: TOtal dimension of the model.
        num_head: Number of attention heads.
        droupout: Dropout rate for attention scroes.
        bias: Whether to include bias in linear layers.

    Inputs:
        sequence: input sequence for self-attention and the query for cross attention
        key_value_state: input for key, values for cross attention
    """

    def __init__(self, d_model, num_head, dropout=0.1, bias=True):
        super().__init__()
        assert d_model%num_head==0, "d_model must be divisible by num_head"

        self.d_model = d_model
        self.num_head = num_head

        self.d_head = d_model//num_head
        self.dropout_rate = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.output_proj = nn.Linear(d_model, d_model, bias=bias)

        #Dropout Layer
        self.dropout = nn.Dropout(p=dropout)

        #Sclaing factor for attention scores
        self.scaler = float(1./math.sqrt(self.d_head))

    def forward(self, sequence, key_value_states=None, att_mask=None):

        batch_size, seq_len, model_dim = sequence.size()

        assert model_dim==self.d_model, f"Input sequence must have dimension {self.d_model}, got {model_dim}"
        if key_value_states is not None:
            assert key_value_states.size(-1)==self.d_model, (f"Cross attention key/value dimension {key_value_states.size(-1)}"
                                                             f"does not match model dimension {self.d_model}")

        is_cross_attention = key_value_states is not None

        Q_state = self.q_proj(sequence)

        if is_cross_attention:
            kv_seq_len = key_value_states.size(1)
            K_state = self.k_proj(key_value_states)
            V_state = self.v_proj(key_value_states)
        else:
            kv_seq_len = seq_len
            K_state = self.k_proj(sequence)
            V_state = self.v_proj(sequence)

        # Reshape Q, K, V states from [batch_size, seq_len, num_head, d_head] to [batch_size, num_head, seq_len, d_head]
        Q_state = Q_state.view(batch_size, seq_len, self.num_head, self.d_head).transpose(1,2)

        K_state = K_state.view(batch_size, kv_seq_len, self.num_head, self.d_head).transpose(1,2)
        V_state = V_state.view(batch_size, kv_seq_len, self.num_head, self.d_head).transpose(1,2)

        # Scaling factor for attention scores. It is same as QKT/sqrt(d_head)
        Q_state = Q_state * self.scaler

        # Q_state: [batch_size, num_head, seq_len, d_head]
        # K_state: [batch_size, num_head, kv_seq_len, d_head] -> Transposed to [batch_size, num_head, d_head, kv_seq_len]
        # Attention matrix is [batch_size, num_head, seq_len, kv_seq_len]
        self.att_matrix = torch.matmul(Q_state, K_state.transpose(-1,-2))

        if att_mask is not None and not isinstance(att_mask, torch.Tensor):
            raise TypeError("att_mask must be a torch.Tensor")

        if att_mask is not None:
            self.att_matrix = self.att_matrix + att_mask

        # Apply softmax to attention matrix to the last dimension to get attention score
        # at each seq_len we now have attention scores summing to 1 for each head
        att_score = F.softmax(self.att_matrix, dim=-1)

        # Apply dropout to attention scores
        att_score = self.dropout(att_score)

        # Multiply attention scores with value states
        # att_score: [batch_size, num_head, seq_len, kv_seq_len]
        # V_state: [batch_size, num_head, kv_seq_len, d_head]
        # Output: [batch_size, num_head, seq_len, d_head]
        att_output = torch.matmul(att_score, V_state)

        # Reshape output from [batch_size, num_head, seq_len, d_head] to [batch_size, seq_len, d_model]
        att_output = att_output.transpose(1,2)
        # concat all the heads together
        att_output = att_output.contiguous().view(batch_size,seq_len,self.num_head*self.d_head)

        # Project the output to d_model dimension
        output = self.output_proj(att_output)

        assert att_output.size() == (batch_size, seq_len, self.d_model), f"Final output shape {att_output.size()} incorrect"

        return att_output