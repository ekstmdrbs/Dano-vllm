import os
from glob import glob
import torch
from torch import nn
from safetensors.torch import load_file as load_safetensors


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    def process_state_dict(state_dict):
        for weight_name, loaded_weight in state_dict.items():
            is_packed = False
            for k in packed_modules_mapping:
                if k in weight_name:
                    v, shard_id = packed_modules_mapping[k]
                    param_name = weight_name.replace(k, v)
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, loaded_weight, shard_id)
                    is_packed = True
                    break
            
            if not is_packed:
                param = model.get_parameter(weight_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

    files = glob(os.path.join(path, "*.safetensors")) + glob(os.path.join(path, "*.bin"))

    for file in files:
        print(f"--> Loading weights from: {file}")
        
        if file.endswith(".safetensors"):
            state_dict = load_safetensors(file, device="cpu")
        elif file.endswith(".bin"):
            state_dict = torch.load(file, map_location="cpu")
        else:
            continue
        
        if hasattr(model, "model") and not any(k.startswith("model.") for k in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith("lm_head"):
                    new_key = "model." + key
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        process_state_dict(state_dict)


        # In nanovllm/utils/loader.py

def load_eagle_model(model: nn.Module, path: str):
    # This loader is specifically for the non-parallel Eagle draft model
    
    # First, load the entire state dict from the checkpoint file(s)
    full_state_dict = {}
    files = glob(os.path.join(path, "*.safetensors")) + glob(os.path.join(path, "*.bin"))
    for file in files:
        print(f"--> Loading weights from: {file}")
        if file.endswith(".safetensors"):
            weights = load_safetensors(file, device="cpu")
        elif file.endswith(".bin"):
            weights = torch.load(file, map_location="cpu", weights_only=True)
        else:
            continue
        full_state_dict.update(weights)

    # Create a new state dict that will match the names and shapes of your Eagle model
    processed_state_dict = {}

    # --- THIS IS THE KEY FIX ---
    # Add the 'model.' prefix to all keys that are not part of the lm_head
    for key, value in full_state_dict.items():
        if not key.startswith("lm_head"):
            new_key = "model." + key
            processed_state_dict[new_key] = value
        else:
            processed_state_dict[key] = value # lm_head keys are kept as is

    # Now, perform the weight concatenation on the correctly prefixed keys
    q_weight = processed_state_dict.pop("model.layers.0.self_attn.q_proj.weight")
    k_weight = processed_state_dict.pop("model.layers.0.self_attn.k_proj.weight")
    v_weight = processed_state_dict.pop("model.layers.0.self_attn.v_proj.weight")
    processed_state_dict["model.layers.0.self_attn.qkv_proj.weight"] = torch.cat(
        [q_weight, k_weight, v_weight], dim=0
    )

    # The MLP weights might need similar handling if they are fused in the checkpoint
    # but separate in your model. Assuming they are separate for now.
    
    # Use the standard PyTorch method to load the final, corrected state dict
    # We set strict=False to ignore any minor mismatches if they exist
    model.load_state_dict(processed_state_dict, strict=False)