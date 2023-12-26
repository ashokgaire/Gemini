import torch
from gemini_torch import Gemini
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Initialize the model
model = Gemini(
    num_tokens=50432,
    max_seq_len=8192,
    dim=2560,
    depth=32,
    dim_head=128,
    heads=24,
    use_abs_pos_emb=False,
    alibi_pos_bias=True,
    alibi_num_heads=12,
    rotary_xpos=True,
    attn_flash=True,
    attn_kv_heads=2,
    qk_norm=True,
    attn_qk_norm=True,
    attn_qk_norm_dim_scale=True,
)

# Initialize the text random tokens
x = torch.randint(0, 5432, (1, 100))
model.to(device)
x = x.to(device)


# Apply model to x
y = model(x)
y = y.cpu()

# Print logits
print(y)

# Print logits
print(y)
