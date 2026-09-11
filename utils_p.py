import ml_collections
from torchvision import transforms

def val_get_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.046, 0.041, 0.030], [0.090, 0.075, 0.065]),
    ])

def get_config():
    config = ml_collections.ConfigDict()

    # 模型参数
    config.hidden_size = 768
    config.classifier = 'token'
    config.eta = 0.2
    config.p = 14

    # Transformer 参数
    config.transformer = ml_collections.ConfigDict()
    config.transformer.num_heads = 12
    config.transformer.mlp_dit_rate = 0.1
    config.transformer.mlp_dim = 3072
    config.transformer.dropouyers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    # Patch 参数
    config.patches = ml_collections.ConfigDict()
    config.patches.size = (16, 16)
    config.representation_size = None
    config.transformer.num_layers = 12
    config.representation_size=None
    return config
