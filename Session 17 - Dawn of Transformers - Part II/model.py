import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable, List

class LayerNormalization(nn.Module):
    """
    Layer normalization module.
    
    Args:
        eps (float): A small value to avoid division by zero.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        x = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return x
    
class FeedForwardBlock(nn.Module):
    """
    Feed-forward block module.
    
    Args:
        d_model (int): Input dimension.
        d_ff (int): Feed-forward dimension.
        dropout (float): Dropout rate.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class InputEmbedding(nn.Module):
    """
    Input embedding module.
    
    Args:
        d_model (int): Dimension of the embedding.
        vocab_size (int): Size of the vocabulary.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for input embedding.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Embedded tensor.
        """
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        return x

class PositionalEmbedding(nn.Module):
    """
    Positional embedding module.
    
    Args:
        d_model (int): Dimension of the embedding.
        seq_len (int): Sequence length.
        dropout (float): Dropout rate.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for positional embedding.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Positional encoded tensor.
        """
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        x = self.dropout(x)
        return x
    
class ResidualConnection(nn.Module):
    """
    Residual connection module with layer normalization.
    
    Args:
        dropout (float): Dropout rate.
    """
    def __init__(self, dropout: float):
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for residual connection.
        
        Args:
            x (torch.Tensor): Input tensor.
            sublayer (Callable[[torch.Tensor], torch.Tensor]): Sublayer function.
        
        Returns:
            torch.Tensor: Output tensor after residual connection.
        """
        x = x + self.dropout(sublayer(self.norm(x)))
        return x

class MultiheadAttentionBlock(nn.Module):
    """
    Multi-head attention block module.
    
    Args:
        d_model (int): Dimension of the model.
        h (int): Number of attention heads.
        dropout (float): Dropout rate.
    """
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h        
        assert d_model % h == 0, "d_model not divisible by h"
        self.d_k = d_model // h
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor], dropout: Optional[nn.Dropout]) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        
        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): Attention mask.
            dropout (Optional[nn.Dropout]): Dropout layer.
        
        Returns:
            torch.Tensor: Attention output tensor.
        """
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, 1e-9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return attention_scores @ value, attention_scores

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for multi-head attention block.
        
        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (Optional[torch.Tensor]): Attention mask.
        
        Returns:
            torch.Tensor: Output tensor after attention.
        """
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)        

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, attention_scores = MultiheadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)
    
class EncoderBlock(nn.Module):
    """
    Encoder block module.
    
    Args:
        attention_block (MultiheadAttentionBlock): Self-attention block.
        feedforward_block (FeedForwardBlock): Feed-forward block.
        dropout (float): Dropout rate.
    """
    def __init__(self, attention_block: MultiheadAttentionBlock, feedforward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = attention_block
        self.feedforward_block = feedforward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for encoder block.
        
        Args:
            x (torch.Tensor): Input tensor.
            src_mask (Optional[torch.Tensor]): Source mask tensor.
        
        Returns:
            torch.Tensor: Output tensor after encoder block.
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feedforward_block)
        return x

class Encoder(nn.Module):
    """
    Encoder module consisting of multiple encoder blocks.
    
    Args:
        layers (List[EncoderBlock]): List of encoder blocks.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the encoder.
        
        Args:
            x (torch.Tensor): Input tensor.
            mask (Optional[torch.Tensor]): Source mask tensor.
        
        Returns:
            torch.Tensor: Output tensor after all encoder blocks and normalization.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    """
    Decoder block module.
    
    Args:
        self_attention_block (MultiheadAttentionBlock): Self-attention block.
        cross_attention_block (MultiheadAttentionBlock): Cross-attention block.
        feedforward_block (FeedForwardBlock): Feed-forward block.
        dropout (float): Dropout rate.
    """
    def __init__(self, self_attention_block: MultiheadAttentionBlock, cross_attention_block: MultiheadAttentionBlock, feedforward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforward_block = feedforward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for decoder block.
        
        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Encoder output tensor.
            src_mask (Optional[torch.Tensor]): Source mask tensor.
            tgt_mask (Optional[torch.Tensor]): Target mask tensor.
        
        Returns:
            torch.Tensor: Output tensor after decoder block.
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feedforward_block)
        return x

class Decoder(nn.Module):
    """
    Decoder module consisting of multiple decoder blocks.
    
    Args:
        layers (List[DecoderBlock]): List of decoder blocks.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the decoder.
        
        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Encoder output tensor.
            src_mask (Optional[torch.Tensor]): Source mask tensor.
            tgt_mask (Optional[torch.Tensor]): Target mask tensor.
        
        Returns:
            torch.Tensor: Output tensor after all decoder blocks and normalization.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    """
    Projection layer module.
    
    Args:
        d_model (int): Dimension of the model.
        vocab_size (int): Size of the vocabulary.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj_layer = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for projection layer.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Log-softmax of projected tensor.
        """
        x = self.proj_layer(x)
        return torch.log_softmax(x, dim=-1)
    
class Transformer(nn.Module):
    """
    Transformer module.
    
    Args:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        src_embed (InputEmbedding): Source embedding layer.
        tgt_embed (InputEmbedding): Target embedding layer.
        src_pos (PositionalEmbedding): Source positional embedding layer.
        tgt_pos (PositionalEmbedding): Target positional embedding layer.
        proj_layer (ProjectionLayer): Projection layer.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEmbedding, tgt_pos: PositionalEmbedding, proj_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Encode the source sequence.
        
        Args:
            src (torch.Tensor): Source tensor.
            src_mask (Optional[torch.Tensor]): Source mask tensor.
        
        Returns:
            torch.Tensor: Encoded source tensor.
        """
        src = self.src_embed(src)        
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return src

    def decode(self, encoder_output: torch.Tensor, src_mask: Optional[torch.Tensor], tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Decode the target sequence.
        
        Args:
            encoder_output (torch.Tensor): Encoder output tensor.
            src_mask (Optional[torch.Tensor]): Source mask tensor.
            tgt (torch.Tensor): Target tensor.
            tgt_mask (Optional[torch.Tensor]): Target mask tensor.
        
        Returns:
            torch.Tensor: Decoded target tensor.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return tgt
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the output tensor to the vocabulary size.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Projected tensor.
        """
        return self.proj_layer(x)

def build_transformer(
    src_vocab_size: int, 
    tgt_vocab_size: int,
    src_seq_len: int, 
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6, 
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048
) -> Transformer:
    """
    Build the transformer model.
    
    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Length of the source sequences.
        tgt_seq_len (int): Length of the target sequences.
        d_model (int, optional): Dimension of the model. Default is 512.
        N (int, optional): Number of layers in encoder and decoder. Default is 6.
        h (int, optional): Number of attention heads. Default is 8.
        dropout (float, optional): Dropout rate. Default is 0.1.
        d_ff (int, optional): Dimension of feed-forward layers. Default is 2048.
    
    Returns:
        Transformer: Transformer model.
    """
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    src_pos = PositionalEmbedding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbedding(d_model, tgt_seq_len, dropout)

    encoder_blocks_l = []
    for _ in range(N):
        self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feedforward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feedforward_block, dropout)
        encoder_blocks_l.append(encoder_block)
    
    decoder_blocks_l = []
    for _ in range(N):
        self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout) 
        feedforward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_block, cross_attention_block, feedforward_block, dropout)
        decoder_blocks_l.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks_l))
    decoder = Decoder(nn.ModuleList(decoder_blocks_l))
    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer