import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForImageClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling


@dataclass
class AttentionConfig:
    num_heads: int
    num_queries: int
    use_norm: bool = True
    use_skip_connection: bool = True
    attention_block: list[str] = field(default_factory=lambda: ["self", "cross"])


class Attention(nn.Module):
    def __init__(
        self,
        out_dim: int,
        use_skip_connection: bool = True,
        use_norm: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.multihead_attn = nn.MultiheadAttention(
            out_dim, num_heads, batch_first=True
        )
        self.use_norm = use_norm
        self.use_skip_connection = use_skip_connection
        if self.use_norm:
            self.norm = nn.LayerNorm(out_dim)

    def forward(self, query, key, value, mask_attention=None):
        attn_output, attn_output_weights = self.multihead_attn(
            query, key, value, attn_mask=mask_attention
        )

        if self.use_skip_connection:
            attn_output = query + attn_output
        if self.use_norm:
            attn_output = self.norm(attn_output)

        return attn_output, attn_output_weights


class SelfAttention(nn.Module):
    def __init__(
        self,
        out_dim: int,
        use_skip_connection: bool = True,
        use_norm: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.attention = Attention(out_dim, use_skip_connection, use_norm, num_heads)

    def forward(
        self, feature: torch.Tensor, mask_attention: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        feature: (B, seq_len, out_dim) tensor
        mask_attention: (B, seq_len) boolean tensor where True means that the token is a padding token
        """
        if mask_attention is not None:
            # NOTE: Convert from (B, seq_len) to (B, seq_len, seq_len) where each positive token
            # can attend to all other positive tokens, and negative tokens can't attend to any
            attn_mask = mask_attention.unsqueeze(1) & mask_attention.unsqueeze(2)

            # WARNING: This is a hack to avoid having nan that broke the training by making all
            # negative tokens interact with each other
            attn_mask2 = (~mask_attention.unsqueeze(1)) & (~mask_attention.unsqueeze(2))

            mask_attention = ~(attn_mask + attn_mask2)

        return self.attention(feature, feature, feature, mask_attention)


class CrossAttention(nn.Module):
    def __init__(
        self,
        out_dim: int,
        use_skip_connection: bool = True,
        use_norm: bool = True,
        num_heads: int = 1,
        num_queries: int = 1,
    ):
        super().__init__()
        self.attention = Attention(out_dim, use_skip_connection, use_norm, num_heads)
        self.num_queries = num_queries
        self.learned_queries = nn.Parameter(torch.randn(num_queries, out_dim))

    def forward(
        self,
        feature: torch.Tensor,
        mask_attention: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        feature: (B, seq_len, out_dim) tensor
        mask_attention: (B, seq_len) boolean tensor where True means that the token is a padding token
        """
        batch_size = feature.size(0)
        learned_queries = self.learned_queries.unsqueeze(0).repeat(batch_size, 1, 1)
        if mask_attention is not None:
            # NOTE: Convert from (B, seq_len) to (B, num_queries, seq_len) by repeating
            # the mask along the num_queries dim
            mask_attention = ~mask_attention.unsqueeze(1).expand(
                -1, self.num_queries, -1
            )

        return self.attention(learned_queries, feature, feature, mask_attention)


class AttentionModule(nn.Module):
    def __init__(self, config: AttentionConfig, out_dim: int):
        super().__init__()
        self.attention_block = config.attention_block
        if "self" in self.attention_block:
            self.self_attention = SelfAttention(
                out_dim,
                num_heads=config.num_heads,
                use_norm=config.use_norm,
                use_skip_connection=config.use_skip_connection,
            )
        if "cross" in self.attention_block:
            self.cross_attention = CrossAttention(
                out_dim,
                num_heads=config.num_heads,
                num_queries=config.num_queries,
                use_norm=config.use_norm,
                use_skip_connection=config.use_skip_connection,
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attention_weights_list = []
        for block in self.attention_block:
            mask_attention = (x != 0).any(dim=-1)

            if block == "self":
                x, attention_weights = self.self_attention(x, mask_attention)
            elif block == "cross":
                x, attention_weights = self.cross_attention(x, mask_attention)
            else:
                raise ValueError(f"Unknown attention block {block}")

            attention_weights_list.append(attention_weights)

        x = x.mean(dim=1)
        return x, attention_weights_list


def extract_mask_features(
    patch_tokens: torch.Tensor, masks: torch.Tensor, use_avgpool: bool = False
):
    masks = masks.to(patch_tokens.device)
    spatial_dim = int(math.sqrt(patch_tokens.shape[1]))
    patch_tokens = patch_tokens.view(
        patch_tokens.shape[0], spatial_dim, spatial_dim, patch_tokens.shape[-1]
    )
    # first, interpolate masks to the same size as patch tokens (we use max pooling to avoid empty masks)
    kernel_size = int(masks.shape[-1] // spatial_dim)
    masks_resized = nn.functional.max_pool2d(
        masks, kernel_size=kernel_size, stride=kernel_size
    )
    masks_resized = masks_resized.permute(
        0, 2, 3, 1
    )  # (bsize, spatial_dim, spatial_dim, 1)
    num_pixels = masks_resized.sum(dim=(1, 2))
    if (num_pixels == 0).any():
        # TODO: fill mask_resized with zeros, or something else?
        raise NotImplementedError()

    if use_avgpool:
        # get patch tokens for mask, with averaging over the spatial dimensions
        mask_tokens = (patch_tokens * masks_resized).sum(
            dim=(1, 2)
        ) / masks_resized.sum(dim=(1, 2))
    else:
        # get patch tokens for mask, with averaging over the spatial dimensions
        mask_tokens = patch_tokens[(masks_resized != 0).squeeze(-1)]

    return mask_tokens


def extract_slice_features(patch_tokens: torch.Tensor, kernel_size: int = 1):
    """
    Input : features of all patches of a whole slice (and not just the features for a mask).
    Input shape : [1, 1024, 768] = [1, nb_of_patches, dim_of_the_embedding]
    Output : Average-pooled features.
    Output shape : [nb_of_patches/kernel_size^2, dim_of_the_embedding]
    """
    spatial_dim = int(math.sqrt(patch_tokens.shape[1]))
    patch_tokens = patch_tokens.view(
        patch_tokens.shape[0], spatial_dim, spatial_dim, patch_tokens.shape[-1]
    )
    stride = kernel_size
    # Perform average pooling
    pooled_tokens = F.avg_pool2d(
        patch_tokens.permute(0, 3, 1, 2),
        kernel_size=kernel_size,
        stride=stride,  # [B, C, H, W]
    ).permute(0, 2, 3, 1)  # [B, H', W', C]
    # Reshape to 2D tensor
    slice_tokens = pooled_tokens.reshape(-1, pooled_tokens.shape[-1])
    return slice_tokens


def pad_features(features: List[torch.Tensor]) -> torch.Tensor:
    biggest_num_token = max([feature.shape[0] for feature in features])

    padded_encoder_features = []
    for feature in features:
        num_tokens = feature.shape[0]
        padding_size = biggest_num_token - num_tokens
        if padding_size > 0:
            padded_feature = F.pad(
                feature, (0, 0, 0, padding_size), mode="constant", value=0
            )
        else:
            padded_feature = feature
        padded_encoder_features.append(padded_feature)
    return torch.stack(padded_encoder_features)


def _add_slice_positional_embedding(features: list[torch.Tensor]) -> list[torch.Tensor]:
    """Create sinusoidal positional embeddings for 3D volume slices.

    Args:
        features: List of tensors, where each tensor has shape (dim) or (tokens_per_slice, dim)
                 representing features for each slice in a 3D volume

    Returns:
        List of tensors with added positional embeddings, maintaining the same shape as input tensors
    """
    num_slices = len(features)
    device = features[0].device
    dim = features[0].shape[-1]
    position = torch.arange(num_slices, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim)
    )

    # Create positional embedding matching input shape
    pos_embed = torch.zeros(num_slices, dim, device=device)
    pos_embed[:, 0::2] = torch.sin(position * div_term)
    pos_embed[:, 1::2] = torch.cos(position * div_term)

    for i in range(num_slices):
        features[i] += pos_embed[i]

    return features


def _get_slice_range(
    full_features: BaseModelOutputWithPooling,
    num_slices: Optional[int],
    mask: Optional[torch.Tensor],
) -> list[int]:
    """Determine which slices to process based on num_slices and mask."""
    assert full_features.last_hidden_state is not None
    if num_slices is not None:
        if mask is not None:
            # Choose the slice as the middle slice where the mask is not zero
            non_empty_slices = torch.nonzero(mask.sum((0, 2, 3)))[:, 0]
            middle_slice = non_empty_slices[(len(non_empty_slices) - 1) // 2]
            middle_slice = middle_slice.item()
        else:
            middle_slice = full_features.last_hidden_state.shape[0] // 2

        start_slice = int(middle_slice) - num_slices // 2
        end_slice = start_slice + num_slices

        slice_range = range(start_slice, end_slice)
    else:  # 'all' mode
        slice_range = range(full_features.last_hidden_state.shape[0])

    # Filter out empty or masked slices
    if mask is not None:
        return [i for i in slice_range if mask[:, i].sum() > 0]
    return [i for i in slice_range if (full_features.last_hidden_state[i] != 0).any()]


def _extract_features_per_slice(
    full_features: BaseModelOutputWithPooling,
    slice_idx: int,
    use_n_blocks: int,
    mask: Optional[torch.Tensor] = None,
    kernel_size: Optional[int] = None,
) -> torch.Tensor:
    """Extract features for a single slice across multiple blocks."""
    features_per_block = []
    for block in range(1, use_n_blocks + 1):
        tokens = full_features.hidden_states[-block][slice_idx, 1:, :].unsqueeze(0)  # type: ignore
        if mask is not None:
            slice_features = extract_mask_features(
                tokens,
                mask[:, slice_idx].unsqueeze(0),
            )
        else:
            if kernel_size is not None:
                slice_features = extract_slice_features(
                    tokens,
                    kernel_size=kernel_size,
                )
            else:
                slice_features = full_features.hidden_states[-block][slice_idx, 1:, :]  # type: ignore
        features_per_block.append(slice_features)
    return torch.cat(features_per_block, dim=-1)


def _process_class_tokens(
    full_features: BaseModelOutputWithPooling,
    slice_range: list[int],
    use_n_blocks: int,
    with_positional_embedding: bool = True,
) -> list[torch.Tensor]:
    """Process class tokens and combine them with features."""
    # Extract class tokens for each slice
    list_class_token_per_slice = []
    for i in slice_range:
        class_token_per_block = [
            full_features.hidden_states[-block][i, 0]
            for block in range(1, use_n_blocks + 1)
        ]  # type: ignore
        class_token_per_slice = torch.cat(class_token_per_block, dim=-1)
        list_class_token_per_slice.append(class_token_per_slice)

    if with_positional_embedding:
        list_class_token_per_slice = _add_slice_positional_embedding(
            list_class_token_per_slice
        )

    return list_class_token_per_slice


def extract_3D_all_features(
    full_features: BaseModelOutputWithPooling,
    mask: Optional[torch.Tensor] = None,
    num_slices: Optional[int] = 5,
    use_class_token: bool = False,
    use_n_blocks: Optional[int] = 1,
    kernel_size: Optional[int] = None,
) -> torch.Tensor:
    """Extract features from a 3D volume."""
    use_n_blocks = 1 if use_n_blocks is None else use_n_blocks
    slice_range = _get_slice_range(full_features, num_slices, mask)

    # Extract features for each slice
    features = [
        _extract_features_per_slice(full_features, i, use_n_blocks, mask, kernel_size)
        for i in slice_range
    ]

    # Add positional embedding
    features = _add_slice_positional_embedding(features)

    # Process class tokens if requested
    if use_class_token:
        class_tokens = _process_class_tokens(
            full_features,
            slice_range,
            use_n_blocks,
        )

        for i in range(len(features)):
            # Add class token to the end of each slice so features go from (num_tokens, dim) to (num_tokens + 1, dim)
            features[i] = torch.cat((class_tokens[i].unsqueeze(0), features[i]), dim=0)

    return torch.cat(
        tuple(features), dim=0
    )  # (total number of token in the mask) x 768


def extract_3D_features_avg_per_slice(
    full_features: BaseModelOutputWithPooling,
    mask: Optional[torch.Tensor] = None,
    num_slices: Optional[int] = 5,
    use_class_token: bool = False,
    use_n_blocks: Optional[int] = 1,
) -> torch.Tensor:
    """Extract features from a 3D volume, and average them per slice."""
    use_n_blocks = 1 if use_n_blocks is None else use_n_blocks
    slice_range = _get_slice_range(full_features, num_slices, mask)

    # Extract features for each slice
    features = [
        _extract_features_per_slice(full_features, i, use_n_blocks, mask).mean(dim=0)
        for i in slice_range
    ]  # List of n tensors (768)

    # Add positional embedding
    features = _add_slice_positional_embedding(features)

    # Process class tokens if requested
    if use_class_token:
        class_tokens = _process_class_tokens(
            full_features,
            slice_range,
            use_n_blocks,
        )
        for i in range(len(features)):
            # Add class token to the end of each slice so features go from (num_tokens, dim) to (num_tokens + 1, dim)
            features[i] = torch.cat((class_tokens[i], features[i]), dim=0)

    return torch.stack(features)  # n x 768


def extract_3D_features_avg_per_volume(
    full_features: BaseModelOutputWithPooling,
    mask: Optional[torch.Tensor] = None,
    num_slices: Optional[int] = 5,
    use_class_token: bool = False,
    use_n_blocks: Optional[int] = 1,
) -> torch.Tensor:
    """Extract features from a 3D volume, and average them across all slices."""
    use_n_blocks = 1 if use_n_blocks is None else use_n_blocks
    slice_range = _get_slice_range(full_features, num_slices, mask)

    # Extract features for each slice
    features = [
        _extract_features_per_slice(full_features, i, use_n_blocks, mask)
        for i in slice_range
    ]
    features = torch.mean(torch.cat(features), dim=0)  # tensor of size 768

    # Process class tokens if requested
    if use_class_token:
        class_tokens = _process_class_tokens(
            full_features,
            slice_range,
            use_n_blocks,
            with_positional_embedding=False,
        )
        class_token = torch.stack(class_tokens).mean(dim=0)
        features = torch.cat((class_token, features), dim=-1)

    return features


def extract_3D_features(
    features: BaseModelOutputWithPooling,
    mask: Optional[torch.Tensor] = None,
    use_avgpool_per_slice: bool = False,
    use_avgpool_on_the_volume: bool = False,
    num_slices: Optional[int] = 5,
    kernel_size: Optional[int] = None,
    use_class_token: bool = False,
    use_n_blocks: Optional[int] = 1,
) -> torch.Tensor:
    if use_avgpool_per_slice:
        features_extracted = extract_3D_features_avg_per_slice(
            features,
            mask,
            num_slices,
            use_class_token,
            use_n_blocks,
        )
    elif use_avgpool_on_the_volume:
        features_extracted = extract_3D_features_avg_per_volume(
            features,
            mask,
            num_slices,
            use_class_token,
            use_n_blocks,
        )
    else:
        features_extracted = extract_3D_all_features(
            features,
            mask,
            num_slices,
            use_class_token,
            use_n_blocks,
            kernel_size,
        )

    return features_extracted


class Dinov2ForImageClassificationConfig(PretrainedConfig):
    model_type = "dinov2_custom_classifier"

    def __init__(
        self,
        model_name: str = "raidium/test-curia",
        num_classes: int = 2,
        num_slices: Optional[int] = None,
        kernel_size: Optional[int] = None,
        use_n_blocks: Optional[int] = 1,
        use_class_token: bool = False,
        use_avgpool_per_slice: bool = False,
        use_avgpool_on_the_volume: bool = False,
        attention_cfg: Optional[dict] = None,
        regression: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_slices = num_slices
        self.kernel_size = kernel_size
        self.use_n_blocks = use_n_blocks
        self.use_class_token = use_class_token
        self.use_avgpool_per_slice = use_avgpool_per_slice
        self.use_avgpool_on_the_volume = use_avgpool_on_the_volume
        self.regression = regression

        self.attention_cfg = attention_cfg
        self.auto_map = {
            "AutoConfig": "modeling_dinov2.Dinov2ForImageClassificationConfig",
            "AutoModelForImageClassification": "modeling_dinov2.Dinov2ForImageClassification",
        }
        self.architectures = ["Dinov2ForImageClassification"]

    def to_dict(self):
        data = super().to_dict()
        data["model_type"] = self.model_type
        data["auto_map"] = self.auto_map
        data["architectures"] = self.architectures
        return data


class Dinov2ForImageClassification(PreTrainedModel):
    config_class = Dinov2ForImageClassificationConfig

    def __init__(self, config: Dinov2ForImageClassificationConfig) -> None:
        super().__init__(config)
        self.dino = AutoModel.from_pretrained(config.model_name)
        self.classifier = nn.Linear(self.dino.config.hidden_size, config.num_classes)
        attention_cfg = config.attention_cfg
        self.num_slices = config.num_slices
        self.kernel_size = config.kernel_size
        self.use_n_blocks = config.use_n_blocks
        self.use_class_token = config.use_class_token
        self.use_avgpool_per_slice = config.use_avgpool_per_slice
        self.use_avgpool_on_the_volume = config.use_avgpool_on_the_volume

        self.attention_module: Optional[AttentionModule] = None
        if attention_cfg is not None:
            attn_config = AttentionConfig(
                num_heads=attention_cfg.get("num_heads", 1),
                num_queries=attention_cfg.get("num_queries", 1),
                use_norm=True,
                use_skip_connection=True,
                attention_block=list(attention_cfg.get("block", ("self", "cross"))),
            )
            self.attention_module = AttentionModule(
                attn_config, self.dino.config.hidden_size
            )

    @staticmethod
    def _slice_positional_embeddings(
        num_slices: int, hidden_dim: int, device: torch.device
    ) -> torch.Tensor:
        positions = torch.arange(num_slices, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, device=device).float()
            * (-math.log(10000.0) / hidden_dim)
        )
        embeddings = torch.zeros(num_slices, hidden_dim, device=device)
        embeddings[:, 0::2] = torch.sin(positions * div_term)
        embeddings[:, 1::2] = torch.cos(positions * div_term)
        return embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **_kwargs,
    ):
        is_3d = pixel_values.ndim == 5

        if not is_3d:
            with torch.no_grad():
                outputs = self.dino(
                    pixel_values=pixel_values, output_hidden_states=True
                )

            cls_tokens = outputs.last_hidden_state[:, 0]
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            if mask is not None:
                mask = mask.float()
                pooled_features = extract_mask_features(
                    patch_tokens, mask, use_avgpool=True
                )
            else:
                pooled_features = cls_tokens
        else:
            if mask is not None:
                mask = mask.float().transpose(3, 1).transpose(2, 3).unsqueeze(1)

            with torch.no_grad():
                encoder_features = []
                for i in range(pixel_values.shape[0]):
                    output = self.dino(
                        pixel_values=pixel_values[i], output_hidden_states=True
                    )

                    if mask is not None:
                        features_extracted = extract_3D_features(
                            output,
                            num_slices=self.num_slices,
                            mask=mask[i],
                            use_avgpool_per_slice=self.use_avgpool_per_slice,
                            use_avgpool_on_the_volume=self.use_avgpool_on_the_volume,
                            use_class_token=self.use_class_token,
                        )
                    else:
                        features_extracted = extract_3D_features(
                            output,
                            num_slices=self.num_slices,
                            kernel_size=self.kernel_size,
                            use_n_blocks=self.use_n_blocks,
                            use_class_token=self.use_class_token,
                            use_avgpool_per_slice=self.use_avgpool_per_slice,
                            use_avgpool_on_the_volume=self.use_avgpool_on_the_volume,
                        )
                    encoder_features.append(
                        features_extracted,
                    )

                pooled_features = pad_features(encoder_features)

            if self.attention_module is not None:
                pooled_features, _ = self.attention_module(pooled_features)

        logits = self.classifier(pooled_features)
        loss = None

        if labels is not None:
            if self.config.regression:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).to(logits.dtype))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.config.num_classes), labels.view(-1)
                )

        return {"loss": loss, "logits": logits}


AutoModelForImageClassification.register(
    Dinov2ForImageClassificationConfig, Dinov2ForImageClassification
)
