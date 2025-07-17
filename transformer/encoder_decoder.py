import torch.nn as nn

from transformer.encoder import TransformerEncoder
from transformer.decoder import TransformerDecoder

class TransformerEncoderDecoder(nn.Module):
    """
    Encoder-Decoder stack of the transformer

    SubLayers:
        TransformerEncoder x 6
        TransformerDecoder x 6

    Args:
        d_model : 512 hidden dimension of the model
        d_embed : embedding dimension, same as d_model in transformer framework
        d_ff : 2048 hidden dimension of the feed forward network
        num_head : 8 number of attention heads
        dropout : 0.1 dropout rate

        bias : True include bias in linear projections
    """

    def __init__(self, num_layer, d_model, d_ff, num_head, dropout=0.1, bias=True):
        super().__init__()

        self.num_layer = num_layer
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_head = num_head
        self.dropout = dropout
        self.bias = bias

        # Encoder and Decoder stacks
        self.encoder_stack = nn.ModuleList([
            TransformerEncoder(d_model, d_ff, num_head, dropout, bias)
            for _ in range(num_layer)
        ])

        self.decoder_stack = nn.ModuleList([
            TransformerDecoder(d_model, d_ff, num_head, dropout, bias)
            for _ in range(num_layer)
        ])

    def forward(self, embed_encoder_input, embed_decoder_input, padding_mask=None):

        encoder_output = embed_encoder_input
        for encoder in self.encoder_stack:
            encoder_output = encoder(encoder_output, padding_mask)

        decoder_output = embed_decoder_input
        for decoder in self.decoder_stack:
            decoder_output = decoder(decoder_output, encoder_output, padding_mask)

        return decoder_output