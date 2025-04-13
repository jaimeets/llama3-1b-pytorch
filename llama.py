#BEAUTIFUL LLAMA!!
import torch
torch.manual_seed(42)
torch.set_default_dtype(torch.bfloat16)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using device', device)

dim = 2048
n_layers = 16
n_heads = 32
n_kv_heads = 8
vocab_size = 128256
norm_eps = 1e-05
rope_theta = 500000.0
context = 1000
batch_size = 1
repeats = n_heads // n_kv_heads
qdim = dim // n_heads
path = '/home/jaimeet/.llama/checkpoints/Llama3.2-1B-Instruct'

mask = torch.tril(torch.ones(1, 1, context, context, device=device))
freqs = rope_theta ** -(torch.arange(0, qdim , 2, device=device, dtype=torch.float32) / (qdim))                                                                                                                                                                                                                                       
freqs_cis = torch.outer(torch.arange(context, device=device,  dtype=torch.float32), freqs)
freqs_cis = torch.polar(torch.ones_like(freqs_cis), freqs_cis)

def load_weights():
    checkpoint = torch.load(path + '/consolidated.00.pth', map_location="cpu", weights_only=True)
    weights={}
    for name, param in checkpoint.items():
        if name.endswith('tok_embeddings.weight') or name.endswith('norm.weight'):
            weights[name] = param
        else:
            weights[name] = param.T
        weights[name] = weights[name].to(device)
    return weights

weights = load_weights()
print('Weights loaded')

def Llama(input):
    x = weights['tok_embeddings.weight'][input]
    for n in range(n_layers):
        x = TransformerBlock(x, n)
    x = RMSNorm(x, weights['norm.weight'])
    out = x @ weights['output.weight']
    return out

def TransformerBlock(input, n):
    input = Attention(RMSNorm(input, weights[f'layers.{n}.attention_norm.weight']), n) + input
    out = FeedForward(RMSNorm(input, weights[f'layers.{n}.ffn_norm.weight']), n) + input
    return out

def Attention(input, n):
    T = input.shape[1] 
    Q = input @ weights[f'layers.{n}.attention.wq.weight']
    K = input @ weights[f'layers.{n}.attention.wk.weight']
    V = input @ weights[f'layers.{n}.attention.wv.weight']

    Q = Q.view(batch_size, T, n_heads, qdim)
    K = K.view(batch_size, T, n_kv_heads, qdim)
    V = V.view(batch_size, T, n_kv_heads, qdim)

    Q_ = torch.view_as_complex(Q.float().reshape(*Q.shape[:-1], -1, 2))
    K_ = torch.view_as_complex(K.float().reshape(*K.shape[:-1], -1, 2))
    shape = [d if i == 1 or i == Q_.ndim - 1 else 1 for i, d in enumerate(Q_.shape)]
    freqs_cis_local = freqs_cis[:T].view(*shape)
    Q_ = torch.view_as_real(Q_ * freqs_cis_local).flatten(3)
    K_ = torch.view_as_real(K_ * freqs_cis_local).flatten(3)
    Q,K = Q_.type_as(Q), K_.type_as(K)

    K = K.repeat_interleave(repeats=repeats, dim=2) 
    V = V.repeat_interleave(repeats=repeats, dim=2)

    Q,K,V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2) 

    weightage = torch.matmul(Q, K.transpose(2,3)) * torch.rsqrt(torch.tensor(qdim).type_as(Q))
    masked = weightage.masked_fill_(mask[:, :, :T, :T]==0, float("-Inf"))
    weightage = torch.softmax(masked.float(), dim=-1).type_as(Q)
    
    out = torch.matmul(weightage, V)
    #flask attention
    # out = torch.nn.functional.scaled_dot_product_attention(Q,K,V, is_causal=True)

    out = out.transpose(1, 2).contiguous().view(batch_size, T, dim)
    out = out @ weights[f'layers.{n}.attention.wo.weight']

    return out

def FeedForward(input, n):
    out = Swish(input @ weights[f'layers.{n}.feed_forward.w1.weight']) * (input @ weights[f'layers.{n}.feed_forward.w3.weight'])
    out = out @ weights[f'layers.{n}.feed_forward.w2.weight']
    return out

def RMSNorm(input, weight): return input * torch.rsqrt(torch.mean(input ** 2, dim=-1, keepdim=True) + norm_eps) * weight
def Sigmoid(input): return 1 / (1 + torch.exp(-input))
def Swish(input): return input * Sigmoid(input)