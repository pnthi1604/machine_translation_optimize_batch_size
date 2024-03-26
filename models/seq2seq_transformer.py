import torch.nn as nn
from torch import Tensor
from .architectures.positional_encoding.sin_cos import PositionalEncoding
from .architectures.word_embedding.input_embedding import InputEmbeddings
from .architectures.transformer.model import Transformer
from .architectures.classifier.projection_layer import ProjectionLayer

class Seq2seqTransformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 pad_id_token,
                 d_model: int=512,
                 num_encoder: int=6,
                 num_decoder: int=8,
                 h: int=8,
                 dropout: float=0.1,
                 d_ff: int=2048,
                 max_len: int=100,
    ):
        super().__init__()
        self.pad_id_token = pad_id_token
        # Transformer
        self.transformer = Transformer(
            d_model=d_model,
            num_decoder=num_decoder,
            num_encoder=num_encoder,
            h=h,
            dropout=dropout,
            d_ff=d_ff
            )
        # Linear
        self.classifier = ProjectionLayer(
            d_model=d_model,
            vocab_size=tgt_vocab_size
        )
        # Src embedding
        self.src_emd = InputEmbeddings(
            d_model=d_model,
            vocab_size=src_vocab_size
        )
        # Tgt embedding
        self.tgt_emd = InputEmbeddings(
            d_model=d_model,
            vocab_size=tgt_vocab_size
        )
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            seq_len=max_len
        )
    
    def encode(self,
               src: Tensor,
               src_mask: Tensor
    ):
        src_emd = self.positional_encoding(self.src_emd(src))
        return self.transformer.encode(src=src_emd, src_mask=src_mask)
    
    def deocde(self,
               encoder_output: Tensor,
               tgt: Tensor,
               tgt_mask: Tensor,
               src_mask: Tensor
    ):
        tgt_emd = self.positional_encoding(self.tgt_emd(tgt))
        return self.transformer.decode(tgt=tgt_emd,
                                       encoder_output=encoder_output,
                                       src_mask=src_mask,
                                       tgt_mask=tgt_mask)

    def out(self, transformer_out: Tensor):
        return self.classifier(transformer_out)

__all__ = [
    "Seq2seqTransformer"
]