import torch
import torch.nn as nn 
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        x = self.embedding(x.long())
        return (x * math.sqrt(self.d_model))
    
# class PositionalEncoding(nn.Module):
#     def __init__(self,
#                 d_model:int,
#                 seq_len:int,
#                 dropout:float)->None:
#         super().__init__()

#         self.d_model = d_model
#         self.seq_len = seq_len
#         self.dropout = nn.Dropout(dropout)
#         # need pos embeddings of (seq_len,d_model)

#         pe = torch.zeros(seq_len,d_model)

#         pos = torch.arange(0,seq_len,dtype = torch.float).unsqueeze(1)
        
#         div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

#         pe[:,0::2] = torch.sin(pos*div)
#         pe[:,1::2] = torch.cos(pos*div)
#         # print(f"\n PE:{pe}\n")
#         pe = pe.unsqueeze(0) #(1,seq_len,d_model)
#         self.register_buffer('pe',pe)
#         print(f"Positional Encoding (PE) shape: {pe.shape}")

#     # def forward(self,x):
#     #     x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
#     #     x = self.dropout(x)
#     #     return x
#     def forward(self, x):
#         # Ensure dimensions match
#         assert x.size(1) <= self.seq_len, f"x.size(1) {x.size(1)} must be <= self.seq_len {self.seq_len}"

#         pe_slice = self.pe[:, :x.shape[1], :]  # Adjust PE to match x's seq_len dimension
#         print(f"PE slice shape: {pe_slice.shape}")

#         x = x + pe_slice.requires_grad_(False)
#         x = self.dropout(x)

#         return x  

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(f"x.shape: {x.shape}")
        # print(f"self.pe.shape: {self.pe.shape}")
        # print(f"x.shape[1]: {x.shape[1]}")
        # print(f"self.pe[:, :x.shape[1], :].shape: {self.pe[:, :x.shape[1], :].shape}")

        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

    
class LayerNorm(nn.Module):
    def __init__(self,eps:float = 1e-6)->None:
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim = -1,keepdim = True)
        std = x.std(dim = -1,keepdim=True)

        x = self.alpha*((x - mean)/(std + self.eps)) + self.beta
        return x
    
class FFN(nn.Module):
    def __init__(self,
                d_model:int = 512,
                dff:int = 2048,
                dropout:float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model,dff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff,d_model)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.dropout(x)
        x = self.linear2(x)

        return x
    
class MultiHeadAttn(nn.Module):

    def __init__(self,
                d_model:int = 512,
                h:int = 8,
                dropout:float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.h = h 
        self.dropout = nn.Dropout(dropout)
        assert d_model%h == 0,"d_model is not divisible by h"

        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)

        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def attn(q,k,v,mask,dp:nn.Dropout):
        d_k = q.shape[-1]
        attn_scores = (q@k.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            attn_scores.masked_fill(mask ==0,-1e10)
        attn_scores = attn_scores.softmax(dim = -1)
        attn_scores = dp(attn_scores)

        return (attn_scores@v),attn_scores

    def forward(self,q,k,v,mask):
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # (batch,seq,dmodel) -> (batch,seq,h,dk) -> (batch,h,seq,dk)
        Q = Q.view(Q.shape[0],Q.shape[1],self.h,self.d_k).transpose(1,2)
        K = K.view(K.shape[0],K.shape[1],self.h,self.d_k).transpose(1,2)
        V = V.view(V.shape[0],V.shape[1],self.h,self.d_k).transpose(1,2)

        x,attn_scores = MultiHeadAttn.attn(Q,K,V,mask,self.dropout)     
        #(batch,h,seq_len,d_k) --> (batch,seq,h,d_k)
        # x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h,self.d_k)
        # x = self.w_o(x)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], self.h * self.d_k)
        x = self.w_o(x)
        return x
    
class ResidualConn(nn.Module):
    def __init__(self,dropout:float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.Lnorm = LayerNorm()

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.Lnorm(x)))
    
class EncoderBLock(nn.Module):

    def __init__(self,self_attn:MultiHeadAttn,ffn:FFN,dp:float):
        super().__init__()

        self.self_attn_block = self_attn
        self.ffn = ffn
        self.residual_conn = nn.ModuleList([
            ResidualConn(dp) for _ in range(2)
        ])

    def forward(self,x,src_mask):
        x = self.residual_conn[0](x,lambda x : self.self_attn_block(x,x,x,src_mask))
        x = self.residual_conn[1](x,lambda x:self.ffn(x))
        
        return x
    
class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers 
        self.norm = LayerNorm()

    def forward(self,x,mask):
        for layers in self.layers:
            x = layers(x,mask)
        return self.norm(x)
    
class Decoderblock(nn.Module):
    def __init__(self, self_attn:MultiHeadAttn,cross_attn:MultiHeadAttn,ffn:FFN,dp:float):
        super().__init__()

        self.self_attn_block = self_attn
        self.cross_attn_block = cross_attn
        self.ffn = ffn 

        self.residual_conn = nn.ModuleList([
            ResidualConn(dp) for _ in range(3)
        ])
    def forward(self, x,encoder_op,src_mask,tgt_mask):
        x = self.residual_conn[0](x,lambda x:self.self_attn_block(x,x,x,tgt_mask))
        x = self.residual_conn[1](x,lambda x:self.cross_attn_block(x,encoder_op,encoder_op,src_mask))
        x = self.residual_conn[2](x,lambda x : self.ffn(x))

        return x
    
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList):
        super().__init__()
        self.layers = layers 
        self.norm = LayerNorm()

    def forward(self,x,encoder_op,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_op,src_mask,tgt_mask)
        return self.norm(x)
    
class ProjLayer(nn.Module):

    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim = -1)
    
class Transformer(nn.Module):

    def __init__(self,encoder:Encoder,
                 decoder:Decoder,
                 src_embed:InputEmbeddings,
                 tgt_embed:InputEmbeddings,
                 src_pos:PositionalEncoding,
                 tgt_pos:PositionalEncoding,
                 projection:ProjLayer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection

    def encode(self,src,src_mask,):
        src = self.src_embed(src)
        src = self.src_pos(src)
        # print(f"\nAfter pos encoder:{src.shape}\n")
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        # print(f"\nAfter dec encoder:{tgt.shape}\n")

        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def proj(self,x):
        return self.projection(x)

def build_transformer(
        src_vocab_size:int,
        tgt_vocab_size:int,
        src_seq_len:int,
        tgt_seq_len:int,
        d_model:int = 512,
        N:int = 6,
        h:int = 8,
        dp:float = 0.1,
        d_ff:int = 2048
):
    src_embed = InputEmbeddings(d_model,src_vocab_size)
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)

    src_pos = PositionalEncoding(d_model,src_seq_len,dp)
    tgt_pos = PositionalEncoding(d_model,tgt_seq_len,dp)

    encoder_blocks,decoder_blocks = [],[]

    for _ in range(N):
        enc_attn = MultiHeadAttn(d_model,h,dp)
        ffn = FFN(d_model,d_ff,dp)

        encoder_block = EncoderBLock(enc_attn,ffn,dp)
        encoder_blocks.append(encoder_block)

    for _ in range(N):
        dec_attn = MultiHeadAttn(d_model,h,dp)
        dec_cross_attn = MultiHeadAttn(d_model,h,dp)
        ffn = FFN(d_model,d_ff,dp)

        dec_block = Decoderblock(dec_attn,dec_cross_attn,ffn,dp)
        decoder_blocks.append(dec_block)

    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projL = ProjLayer(d_model,tgt_vocab_size)

    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projL)

    # Initialize params
    for p in transformer.parameters():
        if p.dim()>1:
           nn.init.xavier_uniform_(p)

    return transformer 