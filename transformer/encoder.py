import torch.nn as nn

from transformer.attention import TransformerAttention
from transformer.feed_forward import FFN

class TransformerEncoder(nn.Module):
    """
    Encoder layer of the transformer

    SubLayers:
        TransformerAttention
        Residual LayerNorm
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

        #Attention Layer
        self.att = TransformerAttention(d_model, num_head, dropout, bias)

        #Feed Forward Network
        self.ffn = FFN(d_model, d_ff)

        #Dropout Layer
        self.dropout = nn.Dropout(p=dropout)

        #layer-norm layers
        self.LayerNorm_att = nn.LayerNorm(d_model)
        self.LayerNorm_ffn = nn.LayerNorm(d_model)

    def forward(self, embed_input, padding_mask=None):

        batch_size, seq_length, d_input = embed_input.size()

        # 1st Sub Layer: Attention
        att_sub_layer = self.att(embed_input, key_value_states=None, att_mask=padding_mask)

        att_sublayer = self.dropout(att_sub_layer)

        # Residual connection and LayerNorm for attention sublayer LayerNorm(x+sublayer(x))
        att_normalized = self.LayerNorm_att(embed_input + att_sublayer)

        # 2nd Sub Layer: Feed Forward Network
        ffn_sub_layer = self.ffn(att_normalized)
        ffn_sublayer = self.dropout(ffn_sub_layer)
        ffn_normalized = self.LayerNorm_ffn(att_normalized + ffn_sublayer)

        return ffn_normalized
