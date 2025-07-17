import math
import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_embed, d_model, max_position_embeddings=512,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.d_embed)
        self.projection = nn.Linear(self.d_embed,self.d_model)

        self.scaling_factor = float(math.sqrt(self.d_model))

        self.layernorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def create_positional_encoding(seq_length, d_model, batch_size=1):

        #Create position indices: [seq_length, 1]
        position = torch.arange(seq_length).unsqueeze(1).float()

        #Create dimension indices: [1, d_model//2]
        #Computes 1/(10000^(2i/d_model)) for i in range(0, d_model, 2)
        div_term = torch.exp(
            torch.arange(0,d_model,2).float() *
            (-math.log(10000.0) / d_model)
        )

        # Create positional encoding tensor: [seq_length, d_model]
        pe = torch.zeros(seq_length, d_model)

        pe[:, 0::2] = torch.sin(position * div_term) # Even Positions Start from 0, +2
        pe[:, 1::2] = torch.cos(position * div_term) # Odd Positions Start from 1, +2

        # Adding dimension and expand for batch_size: [batch_size, seq_length, d_model]
        pe = pe.unsqueeze(0).expand(batch_size,-1,-1)

        return pe

    def forward(self, x):
        assert x.dtype == torch.long, f"Input tensor x must be of type torch.long, got {x.dtype}"

        batch_size, seq_length = x.size()

        #token embedding: [batch_size, seq_length, d_embed]
        token_embedding = self.embedding(x)

        token_embedding = self.projection(token_embedding)*self.scaling_factor

        # add positional encoding
        positional_encoding = self.create_positional_encoding(seq_length, self.d_model, batch_size)

        normalized_sum = self.layernorm(token_embedding + positional_encoding)
        final_output = self.dropout(normalized_sum)

        return final_output