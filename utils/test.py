import torch
from torch.nn import MultiheadAttention

# temp_mha = MultiheadAttention(embed_dim=768, num_heads=1)
# torch.manual_seed(0)
# tweet = torch.rand(1, 9, 768)  # (batch_size, encoder_sequence, d_model)
# replies = torch.rand(1, 9, 768)
# out, attn = temp_mha(value=replies, key=replies, query=tweet)
# pass

from torchnlp.nn import Attention

attention_layer = Attention(dimensions=2, attention_type="general")

query = torch.randn(1, 1, 2)  # tweet
context = torch.randn(1, 3, 2)  # replies
output, weights = attention_layer(query, context)
pass
