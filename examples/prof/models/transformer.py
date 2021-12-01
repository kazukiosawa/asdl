import torch.nn as nn

__all__ = ['transformer_base', 'transformer_big', 'bert_base', 'bert_large',
           'gpt2_base', 'gpt2_medium', 'gpt2_large', 'gpt2_xl']


def transformer_base():
    return nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6,
                          num_decoder_layers=6, dim_feedforward=2048)


def transformer_big():
    return nn.Transformer(d_model=1024, nhead=16, num_encoder_layers=6,
                          num_decoder_layers=6, dim_feedforward=4096)


def bert_base():
    layer = nn.TransformerEncoderLayer(d_model=768, dim_feedforward=3072, nhead=12)
    return nn.TransformerEncoder(layer, num_layers=12)


def bert_large():
    layer = nn.TransformerEncoderLayer(d_model=1024, dim_feedforward=4096, nhead=16)
    return nn.TransformerEncoder(layer, num_layers=24)


def gpt2_base():
    layer = nn.TransformerDecoderLayer(d_model=768, dim_feedforward=3072, nhead=12)
    return nn.TransformerDecoder(layer, num_layers=12)


def gpt2_medium():
    layer = nn.TransformerDecoderLayer(d_model=1024, dim_feedforward=4096, nhead=16)
    return nn.TransformerDecoder(layer, num_layers=24)


def gpt2_large():
    layer = nn.TransformerDecoderLayer(d_model=1280, dim_feedforward=5120, nhead=20)
    return nn.TransformerDecoder(layer, num_layers=36)


def gpt2_xl():
    layer = nn.TransformerDecoderLayer(d_model=1600, dim_feedforward=6400, nhead=25)
    return nn.TransformerDecoder(layer, num_layers=48)
