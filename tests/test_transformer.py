from transformers import AutoTokenizer, pipeline
import torch
import torch.nn.functional as F

from transformer.embedding_positional import Embeddings
from transformer.encoder import TransformerEncoder
from transformer.decoder import TransformerDecoder
from transformer.encoder_decoder import TransformerEncoderDecoder
from transformer.transformer import Transformer

def test_transformer_encoder():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test parameters
    batch_size = 32
    seq_length = 20
    d_model = 512
    d_ff = 2048
    num_heads = 8

    # Initialize the transformer encoder
    encoder = TransformerEncoder(
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_heads,
        dropout=0.1
    )

    # Set to evaluation mode to disable dropout
    encoder.eval()

    # Create input sequence - using ones instead of random values
    # for easier interpretation of attention patterns
    input_sequence = torch.ones(batch_size, seq_length, d_model)
    cross_sequence = torch.ones(batch_size, seq_length, d_model) * 0.5

    # Create attention mask
    attention_mask = torch.ones(batch_size, seq_length)
    attention_mask[:, 15:] = 0  # Mask last 5 positions
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(3)

    # Store attention patterns
    attention_patterns = []

    # Define hook to capture attention scores
    def attention_hook(module, input, output):
        # We want to capture the attention scores before they're processed further
        # This assumes your attention module returns the attention scores
        attention_patterns.append(output)

    # Register the hook on the attention computation
    encoder.att.register_forward_hook(attention_hook)

    # Perform forward pass
    with torch.no_grad():
        output = encoder(input_sequence, attention_mask)

    # Basic shape tests
    expected_shape = (batch_size, seq_length, d_model)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    # Print output statistics
    print("\nOutput Statistics:")
    print(f"Mean: {output.mean():.4f}")
    print(f"Std: {output.std():.4f}")
    print(f"Min: {output.min():.4f}")
    print(f"Max: {output.max():.4f}")

    # Analyze attention patterns
    if attention_patterns:
        attention_output = attention_patterns[0]
        # Look at the attention patterns for unmasked vs masked positions
        unmasked_attention = output[:, :15, :].abs().mean()
        masked_attention = output[:, 15:, :].abs().mean()

        print("\nAttention Analysis:")
        print(f"Unmasked positions mean: {unmasked_attention:.4f}")
        print(f"Masked positions mean: {masked_attention:.4f}")

        # Note: We expect masked positions to still have values due to residual connections,
        # but their patterns should be different from unmasked positions
        print("\nIs the masking working?", "Yes" if unmasked_attention != masked_attention else "No")

    # Check for any NaN or infinite values
    assert torch.isfinite(output).all(), "Output contains NaN or infinite values"

    print("\nAll tests passed successfully!")
    return output, attention_patterns

def test_transformer_decoder():
    torch.manual_seed(42)

    # Test parameters
    batch_size = 32
    seq_length = 20
    encoder_seq_length = 22
    d_model = 512
    d_ff = 2048
    num_heads = 8

    decoder = TransformerDecoder(
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_heads,
        dropout=0.1
    )
    decoder.eval()

    # Create input sequences
    decoder_input = torch.randn(batch_size, seq_length, d_model)
    encoder_output = torch.randn(batch_size, encoder_seq_length, d_model)

    # Create padding mask for encoder outputs
    padding_mask = torch.ones(batch_size, seq_length, encoder_seq_length)
    padding_mask[:, :, 18:] = 0  # Mask last 4 positions of encoder output
    padding_mask = padding_mask.unsqueeze(1)  # Add head dimension

    # Store attention scores
    attention_scores = []

    # Define hook to capture attention scores before softmax
    def attention_hook(module, input, output):
        if not attention_scores:  # Only store first layer's patterns
            # Assuming attention scores are computed before this hook
            attention_scores.append(
                module.att_matrix.detach())  # You might need to modify this based on your attention implementation

    # Register hook on the attention layer
    decoder.att.register_forward_hook(attention_hook)

    # Perform forward pass
    with torch.no_grad():
        output = decoder(decoder_input, encoder_output, padding_mask)

    # Basic shape tests
    expected_shape = (batch_size, seq_length, d_model)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    # Print output statistics
    print("\nOutput Statistics:")
    print(f"Mean: {output.mean():.4f}")
    print(f"Std: {output.std():.4f}")
    print(f"Min: {output.min():.4f}")
    print(f"Max: {output.max():.4f}")

    # Test shape preservation
    print("\nShape Analysis:")
    print(f"Input shape: {decoder_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape matches: {'Yes' if decoder_input.shape == output.shape else 'No'}")

    # Check for any NaN or infinite values
    assert torch.isfinite(output).all(), "Output contains NaN or infinite values"

    print("\nAll tests passed successfully!")
    return output, attention_scores

def test_transformer_encoder_decoder_stack():
    torch.manual_seed(42)

    # Test parameters
    batch_size = 8
    seq_length = 10
    d_model = 512
    d_ff = 2048
    num_heads = 8
    num_layers = 6

    # Initialize the transformer encoder-decoder stack
    transformer = TransformerEncoderDecoder(
        num_layer=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_heads,
        dropout=0.1
    )

    # Set to evaluation mode to disable dropout
    transformer.eval()

    # Create input sequences
    encoder_input = torch.randn(batch_size, seq_length, d_model)
    decoder_input = torch.randn(batch_size, seq_length, d_model)

    # Create padding mask
    padding_mask = torch.ones(batch_size, seq_length)
    padding_mask[:, -2:] = 0  # Mask last 2 positions
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]

    # Store intermediate outputs
    intermediate_outputs = []

    def hook_fn(module, input, output):
        intermediate_outputs.append(output.detach())

    # Register hooks to capture outputs from each encoder and decoder layer
    for i, (encoder, decoder) in enumerate(zip(transformer.encoder_stack, transformer.decoder_stack)):
        encoder.register_forward_hook(lambda m, i, o, layer=i: print(f"\nEncoder Layer {layer} shape:", o.shape))
        decoder.register_forward_hook(lambda m, i, o, layer=i: print(f"Decoder Layer {layer} shape:", o.shape))

    # Perform forward pass
    with torch.no_grad():
        output = transformer(encoder_input, decoder_input, padding_mask)

    # Basic shape tests
    expected_shape = (batch_size, seq_length, d_model)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    # Print output statistics
    print("\nFinal Output Statistics:")
    print(f"Mean: {output.mean():.4f}")
    print(f"Std: {output.std():.4f}")
    print(f"Min: {output.min():.4f}")
    print(f"Max: {output.max():.4f}")

    # Verify shape preservation through layers
    print("\nShape Preservation Check:")
    print(f"Input shapes - Encoder: {encoder_input.shape}, Decoder: {decoder_input.shape}")
    print(f"Output shape: {output.shape}")

    # Check for any NaN or infinite values
    assert torch.isfinite(output).all(), "Output contains NaN or infinite values"

    # Verify that output is different from input (transformation happened)
    input_output_diff = (output - decoder_input).abs().mean()
    print(f"\nMean absolute difference between input and output: {input_output_diff:.4f}")
    print("Transformation occurred:", "Yes" if input_output_diff > 1e-3 else "No")

    # Check if model parameters were used
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"\nTotal number of parameters: {total_params:,}")

    print("\nAll tests passed successfully!")
    return output

def test_complete_transformer():
    # Configuration
    d_model = 768
    d_embed = 1024
    d_ff = 2048
    num_heads = 8
    num_layers = 6
    max_position_embeddings = 512

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",
                                              use_fast=True,
                                              use_multiprocessing=False)
    vocab_size = tokenizer.vocab_size

    # Create sample source and target sequences
    src_sequences = [
        "I've been waiting for a HuggingFace course my whole life.",
        "So have I!"
    ]
    # Pretend these are translations
    tgt_sequences = [
        "J'ai attendu un cours HuggingFace toute ma vie.",
        "Moi aussi!"
    ]

    # Tokenize source and target sequences
    src_inputs = tokenizer(src_sequences, truncation=True, padding="longest", return_tensors="pt")
    tgt_inputs = tokenizer(tgt_sequences, truncation=True, padding="longest", return_tensors="pt")

    # Create transformer model
    transformer = Transformer(
        num_layer=num_layers,
        d_model=d_model,
        d_embed=d_embed,
        d_ff=d_ff,
        num_head=num_heads,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings
    )

    # Set to eval mode
    transformer.eval()

    # Create padding mask from attention mask
    padding_mask = src_inputs['attention_mask'].unsqueeze(1).unsqueeze(2)

    print("\nInput Shapes:")
    print(f"Source tokens: {src_inputs['input_ids'].shape}")
    print(f"Target tokens: {tgt_inputs['input_ids'].shape}")

    # Forward pass
    with torch.no_grad():
        output = transformer(
            src_tokens=src_inputs['input_ids'],
            tgt_tokens=tgt_inputs['input_ids'],
            padding_mask=padding_mask
        )

    print("\nOutput Analysis:")
    print(f"Output shape: {output.shape}")  # Should be [batch_size, tgt_len, vocab_size]

    # Verify output is proper probability distribution
    print("\nProbability Distribution Check:")
    print(f"Sum to 1: {torch.allclose(output.exp().sum(dim=-1), torch.ones_like(output.exp().sum(dim=-1)))}")
    print(f"Max probability: {output.exp().max().item():.4f}")
    print(f"Min probability: {output.exp().min().item():.4f}")

    # Check if we can get predictions
    predictions = output.argmax(dim=-1)
    print("\nSample Predictions:")
    print("Original target:")
    print(tgt_sequences[0])
    print("\nModel output (decoded):")
    print(tokenizer.decode(predictions[0]))

    # Test backward pass
    transformer.train()
    output = transformer(
        src_tokens=src_inputs['input_ids'],
        tgt_tokens=tgt_inputs['input_ids'],
        padding_mask=padding_mask
    )

    # Calculate loss (cross entropy)
    loss = F.nll_loss(
        output.view(-1, vocab_size),
        tgt_inputs['input_ids'].view(-1)
    )

    # Test backward pass
    loss.backward()

    # Verify gradients
    has_gradients = all(p.grad is not None for p in transformer.parameters())
    print("\nTraining Check:")
    print(f"Loss value: {loss.item():.4f}")
    print(f"Has gradients: {has_gradients}")

    return output, predictions

def test_decoder_causal_masking():
    torch.manual_seed(42)

    # Test parameters
    batch_size = 2
    seq_length = 5
    d_model = 512
    d_ff = 2048
    num_heads = 8

    decoder = TransformerDecoder(
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_heads,
        dropout=0.1
    )
    decoder.eval()

    decoder_input = torch.randn(batch_size, seq_length, d_model)
    encoder_output = torch.randn(batch_size, seq_length, d_model)

    attention_scores = []

    def attention_hook(module, input, output):
        if not attention_scores:
            # Apply softmax to get actual attention probabilities
            scores = F.softmax(module.att_matrix, dim=-1)
            attention_scores.append(scores.detach())

    decoder.att.register_forward_hook(attention_hook)

    with torch.no_grad():
        output = decoder(decoder_input, encoder_output)

    att_weights = attention_scores[0]

    print("\nAttention Matrix Shape:", att_weights.shape)

    # Print attention pattern for first head of first batch
    print("\nAttention Pattern (first head):")
    print(att_weights[0, 0].round(decimals=4))

    # Check future tokens (should be 0)
    future_attention = att_weights[:, :, torch.triu_indices(seq_length, seq_length, offset=1)[0],
                       torch.triu_indices(seq_length, seq_length, offset=1)[1]]

    print("\nFuture Token Analysis:")
    print(f"Mean attention to future tokens: {future_attention.mean():.8f}")
    print(f"Max attention to future tokens: {future_attention.max():.8f}")
    print("Causal masking working:", "Yes" if future_attention.mean() < 1e-7 else "No")

    # Check present/past tokens
    present_past = att_weights[:, :, torch.tril_indices(seq_length, seq_length)[0],
                   torch.tril_indices(seq_length, seq_length)[1]]

    print("\nPresent/Past Token Analysis:")
    print(f"Mean attention to present/past tokens: {present_past.mean():.4f}")
    print(f"Has non-zero attention patterns:", "Yes" if present_past.mean() > 0 else "No")

    # Verify each position's attention sums to 1
    attention_sums = att_weights.sum(dim=-1)
    print("\nAttention Sum Analysis:")
    print(f"Mean attention sum (should be 1): {attention_sums.mean():.4f}")
    print(f"Max deviation from 1: {(attention_sums - 1).abs().max():.8f}")

    return att_weights


def test_decoder_cross_attention():
    torch.manual_seed(42)

    # Test parameters
    batch_size = 2
    decoder_seq_len = 5
    encoder_seq_len = 7  # Different length to make it interesting!
    d_model = 512
    d_ff = 2048
    num_heads = 8

    decoder = TransformerDecoder(
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_heads,
        dropout=0.1
    )
    decoder.eval()

    # Create input sequences
    decoder_input = torch.randn(batch_size, decoder_seq_len, d_model)
    encoder_output = torch.randn(batch_size, encoder_seq_len, d_model)

    # Store attention scores
    cross_attention_scores = []

    def attention_hook(module, input, output):
        # We want the second call to att (cross-attention)
        if len(cross_attention_scores) < 2:
            scores = F.softmax(module.att_matrix, dim=-1)
            cross_attention_scores.append(scores.detach())

    decoder.att.register_forward_hook(attention_hook)

    # Forward pass
    with torch.no_grad():
        output = decoder(decoder_input, encoder_output)

    # Get cross-attention weights (second element in list)
    cross_att_weights = cross_attention_scores[1]  # [batch, heads, decoder_seq_len, encoder_seq_len]

    print("\nCross-Attention Matrix Shape:", cross_att_weights.shape)

    # Print attention pattern for first head of first batch
    print("\nCross-Attention Pattern (first head):")
    print(cross_att_weights[0, 0].round(decimals=4))

    # Verify each decoder position attends to all encoder positions
    attention_sums = cross_att_weights.sum(dim=-1)
    zero_attention = (cross_att_weights == 0).all(dim=-1)

    print("\nCross-Attention Analysis:")
    print(f"Mean attention weight: {cross_att_weights.mean():.4f}")
    print(f"Min attention weight: {cross_att_weights.min():.4f}")
    print(f"Max attention weight: {cross_att_weights.max():.4f}")

    print("\nAttention Coverage:")
    print(f"Each position's attention sums to 1: {torch.allclose(attention_sums, torch.ones_like(attention_sums))}")
    print(f"Every decoder position attends to some encoder position: {not zero_attention.any()}")

    # Check attention distribution
    attention_entropy = -(cross_att_weights * torch.log(cross_att_weights + 1e-9)).sum(dim=-1).mean()
    print(f"\nAttention entropy (higher means more uniform attention): {attention_entropy:.4f}")

    return cross_att_weights


def test_decoder_cross_attention_with_padding():
    torch.manual_seed(42)

    # Test parameters
    batch_size = 2
    decoder_seq_len = 5
    encoder_seq_len = 7
    d_model = 512
    d_ff = 2048
    num_heads = 8

    decoder = TransformerDecoder(
        d_model=d_model,
        d_ff=d_ff,
        num_head=num_heads,
        dropout=0.1
    )
    decoder.eval()

    # Create input sequences
    decoder_input = torch.randn(batch_size, decoder_seq_len, d_model)
    encoder_output = torch.randn(batch_size, encoder_seq_len, d_model)

    # Create padding mask for encoder outputs
    # Mask out last 2 positions (as if they were padding in encoder output)
    padding_mask = torch.ones(batch_size, decoder_seq_len, encoder_seq_len)
    padding_mask[:, :, -2:] = float('-inf')  # Mask positions 5,6
    padding_mask = padding_mask.unsqueeze(1)  # Add head dimension [batch, 1, decoder_seq, encoder_seq]

    cross_attention_scores = []

    def attention_hook(module, input, output):
        if len(cross_attention_scores) < 2:
            scores = F.softmax(module.att_matrix, dim=-1)
            cross_attention_scores.append(scores.detach())

    decoder.att.register_forward_hook(attention_hook)

    # Forward pass
    with torch.no_grad():
        output = decoder(decoder_input, encoder_output, padding_mask)

    # Get cross-attention weights (second element)
    cross_att_weights = cross_attention_scores[1]

    print("\nCross-Attention Matrix Shape:", cross_att_weights.shape)

    print("\nCross-Attention Pattern (first head):")
    print("(Last two encoder positions should have zero attention)")
    print(cross_att_weights[0, 0].round(decimals=4))

    # Analyze masked positions (last two columns)
    masked_attention = cross_att_weights[:, :, :, -2:]
    unmasked_attention = cross_att_weights[:, :, :, :-2]

    print("\nMasking Analysis:")
    print(f"Mean attention to masked positions: {masked_attention.mean():.8f}")
    print(f"Max attention to masked positions: {masked_attention.max():.8f}")
    print(f"Mean attention to unmasked positions: {unmasked_attention.mean():.4f}")

    # Verify attention still sums to 1 (only over unmasked positions)
    attention_sums = cross_att_weights.sum(dim=-1)

    print("\nAttention Coverage:")
    print(
        f"Each position's attention sums to 1: {torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-6)}")

    # Analyze attention distribution over unmasked positions
    print("\nUnmasked Position Analysis:")
    print(f"Min attention to unmasked positions: {unmasked_attention.min():.4f}")
    print(f"Max attention to unmasked positions: {unmasked_attention.max():.4f}")

    return cross_att_weights

def main():

    d_model = 512
    d_embed = 1024
    vocab_size = 30522

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, use_multiprocessing=False)
    sequences = ["This is my implementation of Attention is All you need Paper",
                 "This will build my understanding on transformers and LLMs"]

    max_position_embeddings = 512
    model_inputs = tokenizer(sequences, truncation=True, padding="longest")

    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocabulary size: {vocab_size}")

    input = torch.tensor(model_inputs["input_ids"])
    embedder = Embeddings(vocab_size, d_embed, d_model)
    output = embedder(input)

    print(f"Input shape: {input.shape}")
    print(f"Embedded shape after projection: {output.shape}")

    # Run the test
    output, attention_patterns = test_transformer_encoder()

    # Run the test
    output, attention_scores = test_transformer_decoder()

    # Run the test
    output = test_transformer_encoder_decoder_stack()

    # Run test
    output, predictions = test_complete_transformer()

    attention_weights = test_decoder_causal_masking()

    # Run the test
    cross_attention_weights = test_decoder_cross_attention()

    # Run the test
    cross_attention_weights = test_decoder_cross_attention_with_padding()

if __name__ == "__main__":
    main()