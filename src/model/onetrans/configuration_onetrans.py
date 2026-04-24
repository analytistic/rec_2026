from transformers import PretrainedConfig

class OneTransConfig(PretrainedConfig):
    model_type = "onetrans"

    def __init__(
            self,
            seq_dim=512,
            
    )