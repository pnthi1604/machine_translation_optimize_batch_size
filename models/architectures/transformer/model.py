import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .multi_head_attention_block import MultiHeadAttentionBlock
from .feed_forward_block import FeedForwardBlock
from .encoder_block import EncoderBlock
from .decoder_block import DecoderBlock


class Transformer(nn.Module):
    def __init__(
            self,
            d_model: int=512, 
            num_encoder: int=6, 
            num_decoder: int=6,
            h: int=8, 
            dropout: float=0.1, 
            d_ff: int=2048,
            custom_encoder=None,
            custom_decoder=None
    ):
        super().__init__()

        encoder_blocks = []
        for _ in range(num_encoder):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = custom_encoder
            if not custom_encoder:
                encoder_block = EncoderBlock(d_model, 
                                             encoder_self_attention_block, 
                                             feed_forward_block, 
                                             dropout)
            encoder_blocks.append(encoder_block)

        decoder_blocks = []
        for _ in range(num_decoder):
            decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            decoder_block = custom_decoder
            if not custom_decoder:
                decoder_block = DecoderBlock(d_model, 
                                             decoder_self_attention_block, 
                                             decoder_cross_attention_block, 
                                             feed_forward_block, 
                                             dropout)
            decoder_blocks.append(decoder_block)

        encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

        self.encoder = encoder
        self.decoder = decoder
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
__all__= [
    "Transformer"
]
