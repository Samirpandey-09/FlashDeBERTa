import torch
import time
from flashdeberta.model import FlashDisentangledSelfAttention
from transformers.models.deberta_v2.modeling_deberta_v2 import DisentangledSelfAttention

class DummyConfig:
    def __init__(self, hidden_size, num_attention_heads, position_buckets, max_relative_positions, pos_att_type=[], max_position_embeddings=512):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1
        # For testing, use both position attention types.
        self.pos_att_type = pos_att_type
        self.relative_attention = True
        self.position_buckets = position_buckets
        self.max_relative_positions = max_relative_positions
        self.max_position_embeddings = max_position_embeddings
        self.share_att_key = False

def compare_flash_and_deberta(B, L, hidden_size, causal=False, sm_scale=None,
                              position_buckets=32, max_relative_positions=64, pos_att_type=[]):
    """
    Compares outputs between the flash implementation and the original Deberta module,
    enforcing the same weights. Returns the mean absolute difference between outputs.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create dummy hidden states.
    hidden_states = torch.randn(B, L, hidden_size, device=device)

    attention_mask = torch.ones(B, L, device=device)

    # Extended attention mask for the original Deberta model.
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)

    # Create random relative positional embeddings.
    rel_embeddings = torch.randn(max_relative_positions, hidden_size, device=device)

    # For testing, choose a fixed number of attention heads.
    num_attention_heads = 8
    assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

    # Instantiate a dummy configuration.
    config = DummyConfig(hidden_size=hidden_size,
                         num_attention_heads=num_attention_heads,
                         position_buckets=position_buckets,
                         max_relative_positions=max_relative_positions,
                         pos_att_type=pos_att_type)

    # Instantiate the original and flash models.
    deberta_model = DisentangledSelfAttention(config).to(device)
    flash_model = FlashDisentangledSelfAttention(config).to(device)

    # Set both models to evaluation mode to disable dropout.
    deberta_model.eval()
    flash_model.eval()

    # Copy weights from the original model to the flash version.
    flash_model.load_state_dict(deberta_model.state_dict())

    # Run a forward pass with the original Deberta model.
    t_start = time.time()
    output_deberta = deberta_model(hidden_states, extended_attention_mask, rel_embeddings=rel_embeddings)
    t_deberta = 1000 * (time.time() - t_start)
    print('Deberta forward pass time (ms):', t_deberta)
    print('Deberta output sample:', output_deberta[0])  # Slice printed for brevity

    # Run a forward pass with the Flash model.
    t_start = time.time()
    output_flash = flash_model(hidden_states, attention_mask, rel_embeddings=rel_embeddings)
    t_flash = 1000 * (time.time() - t_start)
    print('Flash forward pass time (ms):', t_flash)
    print('Flash output sample:', output_flash[0])  # Slice printed for brevity

    # Compute the mean absolute difference between outputs.
    diff = (output_deberta[0] - output_flash[0]).abs().mean().item()
    print("Mean absolute difference between outputs:", diff)
    return diff

def test_flash_vs_deberta():
    """
    This test compares the outputs of the flash and original Deberta attention modules.
    The test passes if the mean absolute difference is below the tolerance threshold.
    """
    # Test parameters.
    B = 1                       # Batch size
    L = 2048                    # Sequence length
    hidden_size = 1024          # hidden_size should be divisible by num_attention_heads (8 here)
    causal = False
    sm_scale = None
    position_buckets = 256
    max_relative_positions = 512
    pos_att_type = ['c2p', 'p2c']  # Example position attention types

    # Compute the difference between outputs.
    diff = compare_flash_and_deberta(
        B, L, hidden_size, causal, sm_scale, position_buckets, max_relative_positions, pos_att_type
    )

    # Define a tolerance threshold.
    tolerance = 1e-4
    # The test will fail if the outputs differ by more than the tolerated threshold.
    assert diff < tolerance, f"Difference {diff} exceeds tolerance {tolerance}"
