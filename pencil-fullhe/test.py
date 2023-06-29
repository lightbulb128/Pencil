import torch.nn as nn
import torch
import torch_models

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
def convertBertOutput():
    loaded_model = load_from_file(..)
    converted_model = torch.nn.Sequential(
        torch_models.Residual(
            torch.nn.Sequential(
                loaded_model.dense,
                loaded_model.dropout
            )
        ),
        loaded_model.LayerNorm
    )
    return converted_model

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
def convertBertIntermediate(loaded_model: BertIntermediate):
    converted_model = torch.nn.Sequential(
        BertIntermediate.dense,
        torch.nn.ReLU()
    )
    return converted_model

def test(input_shape, loaded_model):
    x = torch.randn(input_shape)
    y1 = loaded_model(x)
    y2 = convertBertIntermediate(loaded_model)(x)
    absmax = torch.max(torch.abs(y1 - y2))
    print(absmax)

    torch.nn.TransformerEncoderLayer

