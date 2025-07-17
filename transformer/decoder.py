import torch
import torch.nn as nn

from transformer.attention import TransformerAttention
from transformer.feed_forward import FFN

class TransformerDecoder(nn.Module):
    """
    Decoder Layer of the Transformer

    SubLayers:
        TransformerAttention with Self-Attention
        Residual LayerNorm
        TransformerAttention with Cross-Attention
        FeedForwardNetwork
        Residual LayerNorm

    Args:
        d_model : 512 hidden dimension of the model
        d_embed : embedding dimension, same as d_model in transformer framework
        d_ff : 2048 hidden dimension of the feed forward network
        num_head : 8 number of attention heads
        dropout : 0.1 dropout rate

        bias : True include bias in linear projections
    """

    def __init__(self, d_model, d_ff, num_head, dropout=0.1, bias=True):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # Attention Layer
        self.att = TransformerAttention(d_model, num_head, dropout, bias)

        # FFN Layer
        self.ffn = FFN(d_model, d_ff)

        # dropout Layer
        self.dropout = nn.Dropout(p=dropout)

        # LayerNorm layers
        self.LayerNorm_self_att = nn.LayerNorm(d_model)
        self.LayerNorm_cross_att = nn.LayerNorm(d_model)
        self.LayerNorm_ffn = nn.LayerNorm(d_model)

    @staticmethod
    def create_causal_mask(seq_length):
        """
        Create a causal mask for the self-attention mechanism.

        Args:
            seq_length : Length of the sequence to mask
        Returns:
            causal_mask : Causal mask tensor
        """

        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def forward(self, embed_input, cross_input, padding_mask=None):
        """
        :param embed_input: [batch_size, seq_len, d_model]
        :param cross_input: [batch_size, encoder_seq_len, d_model]
        :param padding_mask: [batch_size, seq_len, encoder_seq_len]
        :return: Tensor of shape [batch_size, seq_len, d_model]
        """

        batch_size, seq_len, _ = embed_input.size()

        assert embed_input.size(-1) == self.d_model, f"Input dimension {embed_input.size(-1)} doesn't match d_model {self.d_model}"
        assert cross_input.size(-1) == self.d_model, f"Cross input dimension {cross_input.size(-1)} doesn't match d_model {self.d_model}"

        # Generate causal mask for self-attention
        causal_mask = self.create_causal_mask(seq_len).to(embed_input.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]

        # 1st Sub Layer: Self-Attention
        att_sub_layer1 = self.att(embed_input, key_value_states=None, att_mask=causal_mask)
        att_sublayer1 = self.dropout(att_sub_layer1)
        att_normalized1 = self.LayerNorm_self_att(embed_input + att_sublayer1)

        # 2nd Sub Layer: Cross-Attention
        att_sub_layer2 = self.att(att_normalized1, key_value_states=cross_input, att_mask=padding_mask)
        att_sublayer2 = self.dropout(att_sub_layer2)
        att_normalized2 = self.LayerNorm_cross_att(att_normalized1 + att_sublayer2)

        # 3rd Sub Layer: Feed Forward Network
        ffn_sub_layer = self.ffn(att_normalized2)
        ffn_sublayer = self.dropout(ffn_sub_layer)
        ffn_normalized = self.LayerNorm_ffn(att_normalized2 + ffn_sublayer)

        return ffn_normalized