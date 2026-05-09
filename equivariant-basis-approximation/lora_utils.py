import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer with LoRA.

    Output:
        y = original_linear(x) + scaling * B(A(x))

    Original linear layer is frozen.
    Only LoRA A and B are trainable.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear modules.")

        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze original layer
        for param in self.base.parameters():
            param.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # LoRA initialization
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


def _set_child_module(parent: nn.Module, child_name: str, new_module: nn.Module):
    setattr(parent, child_name, new_module)


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target: str = "attn",
):
    """
    Applies LoRA to TokenGT linear layers.

    target = "attn":
        applies LoRA to attention layers:
        fc1, fc_v, fc_out

    target = "all":
        applies LoRA to attention + FFN linear layers.
    """

    adapted = []

    for module_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue

            full_name = f"{module_name}.{child_name}" if module_name else child_name

            is_attn_linear = (
                full_name.endswith("attn.fc1")
                or full_name.endswith("attn.fc_v")
                or full_name.endswith("attn.fc_out")
            )

            is_ffn_linear = "ffn" in full_name

            should_adapt = False

            if target == "attn" and is_attn_linear:
                should_adapt = True
            elif target == "all" and (is_attn_linear or is_ffn_linear):
                should_adapt = True

            if should_adapt:
                wrapped = LoRALinear(
                    base_linear=child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                _set_child_module(module, child_name, wrapped)
                adapted.append(full_name)

    return adapted


def freeze_non_lora_params(model: nn.Module):
    """
    Freezes every parameter except LoRA parameters.
    """

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def print_trainable_params(model: nn.Module):
    trainable = 0
    total = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    pct = 100 * trainable / total if total > 0 else 0

    print("=" * 70)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable percent:   {pct:.4f}%")
    print("=" * 70)