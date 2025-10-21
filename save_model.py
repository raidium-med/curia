# build_hf_from_sd.py
import argparse
import re
from collections import OrderedDict

import torch
from omegaconf import OmegaConf
from transformers.models.dinov2 import Dinov2Config, Dinov2Model


def convert_repo_sd_to_hf(state_dict: dict, force_channels: int | None = None):
    # unwrap common wrappers
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "student" in state_dict:
        state_dict = state_dict["student"]

    # your two lines (prefix cleanup)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    # drop DINO/IBOT heads
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not (k.startswith("dino_head.") or k.startswith("ibot_head."))
    }

    hf_sd = OrderedDict()

    # ---- Embeddings / patch embed / tokens ----
    def copy(src, dst):
        if src in state_dict:
            hf_sd[dst] = state_dict[src]

    copy("cls_token", "embeddings.cls_token")
    copy("pos_embed", "embeddings.position_embeddings")
    copy("mask_token", "embeddings.mask_token")
    copy("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight")
    copy("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias")

    # ---- Encoder blocks ----
    # repo: blocks.<stage>.<global_idx>.<...>
    block_pat = re.compile(r"^blocks\.(\d+)\.(\d+)\.(.+)$")
    per_block = {}
    for k, v in state_dict.items():
        m = block_pat.match(k)
        if not m:
            continue
        gidx = int(m.group(2))  # global block index (0..num_layers-1)
        tail = m.group(3)
        per_block.setdefault(gidx, {})[tail] = v

    # infer hidden_size from qkv
    hidden_size = None
    for parts in per_block.values():
        if "attn.qkv.weight" in parts:
            hidden_size = parts["attn.qkv.weight"].shape[0] // 3
            break

    for i, parts in sorted(per_block.items()):
        # norm1
        if "norm1.weight" in parts:
            hf_sd[f"encoder.layer.{i}.norm1.weight"] = parts["norm1.weight"]
        if "norm1.bias" in parts:
            hf_sd[f"encoder.layer.{i}.norm1.bias"] = parts["norm1.bias"]

        # attention qkv split
        if "attn.qkv.weight" in parts and hidden_size is not None:
            qkv_w = parts["attn.qkv.weight"]
            hf_sd[f"encoder.layer.{i}.attention.attention.query.weight"] = qkv_w[
                0:hidden_size, :
            ]
            hf_sd[f"encoder.layer.{i}.attention.attention.key.weight"] = qkv_w[
                hidden_size : 2 * hidden_size, :
            ]
            hf_sd[f"encoder.layer.{i}.attention.attention.value.weight"] = qkv_w[
                2 * hidden_size :, :
            ]

        if "attn.qkv.bias" in parts and hidden_size is not None:
            qkv_b = parts["attn.qkv.bias"]
            hf_sd[f"encoder.layer.{i}.attention.attention.query.bias"] = qkv_b[
                0:hidden_size
            ]
            hf_sd[f"encoder.layer.{i}.attention.attention.key.bias"] = qkv_b[
                hidden_size : 2 * hidden_size
            ]
            hf_sd[f"encoder.layer.{i}.attention.attention.value.bias"] = qkv_b[
                2 * hidden_size :
            ]

        # attn out proj
        if "attn.proj.weight" in parts:
            hf_sd[f"encoder.layer.{i}.attention.output.dense.weight"] = parts[
                "attn.proj.weight"
            ]
        if "attn.proj.bias" in parts:
            hf_sd[f"encoder.layer.{i}.attention.output.dense.bias"] = parts[
                "attn.proj.bias"
            ]

        # layer scales
        if "ls1.gamma" in parts:
            hf_sd[f"encoder.layer.{i}.layer_scale1.lambda1"] = parts["ls1.gamma"]
        if "ls2.gamma" in parts:
            hf_sd[f"encoder.layer.{i}.layer_scale2.lambda1"] = parts["ls2.gamma"]

        # norm2
        if "norm2.weight" in parts:
            hf_sd[f"encoder.layer.{i}.norm2.weight"] = parts["norm2.weight"]
        if "norm2.bias" in parts:
            hf_sd[f"encoder.layer.{i}.norm2.bias"] = parts["norm2.bias"]

        # MLP (w12 -> weights_in, w3 -> weights_out)
        if "mlp.w12.weight" in parts:
            hf_sd[f"encoder.layer.{i}.mlp.weights_in.weight"] = parts["mlp.w12.weight"]
        if "mlp.w12.bias" in parts:
            hf_sd[f"encoder.layer.{i}.mlp.weights_in.bias"] = parts["mlp.w12.bias"]
        if "mlp.w3.weight" in parts:
            hf_sd[f"encoder.layer.{i}.mlp.weights_out.weight"] = parts["mlp.w3.weight"]
        if "mlp.w3.bias" in parts:
            hf_sd[f"encoder.layer.{i}.mlp.weights_out.bias"] = parts["mlp.w3.bias"]

    # ---- Final LayerNorm ----
    copy("norm.weight", "layernorm.weight")
    copy("norm.bias", "layernorm.bias")

    # ---- Optional: force to 1 channel by averaging RGB ----
    if force_channels == 1:
        wkey = "embeddings.patch_embeddings.projection.weight"
        if wkey in hf_sd and hf_sd[wkey].shape[1] != 1:
            hf_sd[wkey] = hf_sd[wkey].mean(dim=1, keepdim=True)

    num_layers = (max(per_block.keys()) + 1) if per_block else 0
    return hf_sd, hidden_size, num_layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_id", type=str, default=None, help="Run ID to load config from"
    )
    parser.add_argument(
        "--step", type=int, default=None, help="Checkpoint step to load"
    )
    args = parser.parse_args()
    conf = OmegaConf.load(f"experiments/{args.run_id}/config.yaml")

    cfg = Dinov2Config(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        initializer_range=0.02,
        layer_norm_eps=0.000001,
        image_size=conf.crops["global_crops_size"],
        patch_size=conf.student["patch_size"],
        num_channels=conf.student["in_chans"],
        qkv_bias=conf.student["qkv_bias"],
        layerscale_value=conf.student["layerscale"],
        drop_path_rate=conf.student["drop_path_rate"],
        use_swiglu_ffn=conf.student["ffn_layer"] == "swiglufused",
        out_features=None,
        out_indices=None,
        apply_layernorm=True,
        reshape_hidden_states=True,
        use_mask_token=True,
    )

    model = Dinov2Model(cfg)
    sd = torch.load(
        f"experiments/{args.run_id}/eval/training_{args.step}/teacher_checkpoint.pth",
        map_location="cpu",
    )
    hf_sd, hidden_size, num_layers = convert_repo_sd_to_hf(sd["teacher"])

    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    print("missing:", missing, "\nunexpected:", unexpected)
    #
    model.save_pretrained("./curia", safe_serialization=True)
    cfg.save_pretrained("./curia")
