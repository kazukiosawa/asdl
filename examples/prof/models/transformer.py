import torch.nn as nn

__all__ = ['transformer_base', 'transformer_big', 'bert_base', 'bert_large',
           'gpt2_base', 'gpt2_medium', 'gpt2_large', 'gpt2_xl']


def transformer_base(num_encoder_layers=6, num_decoder_layers=6):
    return nn.Transformer(d_model=512, nhead=8, num_encoder_layers=num_encoder_layers,
                          num_decoder_layers=num_decoder_layers, dim_feedforward=2048)


def transformer_big(num_encoder_layers=6, num_decoder_layers=6):
    return nn.Transformer(d_model=1024, nhead=16, num_encoder_layers=num_encoder_layers,
                          num_decoder_layers=num_decoder_layers, dim_feedforward=4096)


def bert_base(num_layers=12):
    layer = nn.TransformerEncoderLayer(d_model=768, dim_feedforward=3072, nhead=12)
    return nn.TransformerEncoder(layer, num_layers=num_layers)


def bert_large(num_layers=24):
    layer = nn.TransformerEncoderLayer(d_model=1024, dim_feedforward=4096, nhead=16)
    return nn.TransformerEncoder(layer, num_layers=num_layers)


def gpt2_base(num_layers=12):
    layer = nn.TransformerDecoderLayer(d_model=768, dim_feedforward=3072, nhead=12)
    return nn.TransformerDecoder(layer, num_layers=num_layers)


def gpt2_medium(num_layers=24):
    layer = nn.TransformerDecoderLayer(d_model=1024, dim_feedforward=4096, nhead=16)
    return nn.TransformerDecoder(layer, num_layers=num_layers)


def gpt2_large(num_layers=36):
    layer = nn.TransformerDecoderLayer(d_model=1280, dim_feedforward=5120, nhead=20)
    return nn.TransformerDecoder(layer, num_layers=num_layers)


def gpt2_xl(num_layers=48):
    layer = nn.TransformerDecoderLayer(d_model=1600, dim_feedforward=6400, nhead=25)
    return nn.TransformerDecoder(layer, num_layers=num_layers)
