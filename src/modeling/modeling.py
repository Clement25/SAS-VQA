"""
Transformer part of ClipBERT
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformers import CLIPTextModel, CLIPVisionModelWithProjection
from transformers import BlipForQuestionAnswering
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm


def get_random_sample_indices(
        seq_len, num_samples=100, device=torch.device("cpu")):
    """
    Args:
        seq_len: int, the sampled indices will be in the range [0, seq_len-1]
        num_samples: sample size
        device: torch.device

    Returns:
        1D torch.LongTensor consisting of sorted sample indices
        (sort should not affect the results as we use transformers)
    """
    if num_samples >= seq_len:
        # return all indices
        sample_indices = np.arange(seq_len)
    else:
        sample_indices = np.random.choice(
            seq_len, size=num_samples, replace=False)
        sample_indices = np.sort(sample_indices)
    return torch.from_numpy(sample_indices).long().to(device)

BertLayerNorm = LayerNorm

class VisualInputEmbedding(nn.Module):
    """
    Takes input of both image and video (multi-frame)
    """
    def __init__(self, config):
        super(VisualInputEmbedding, self).__init__()
        self.config = config

        # sequence embedding
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.row_position_embeddings = nn.Embedding(
            config.max_grid_row_position_embeddings,
            config.hidden_size)
        self.col_position_embeddings = nn.Embedding(
            config.max_grid_col_position_embeddings,
            config.hidden_size)
        self.token_type_embeddings = nn.Embedding(1, config.hidden_size)
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, grid):
        """
        Args:
            grid: (B, n_frm, H, W, C), note that #frm can be 1

        Returns:

        """
        bsz, _, _, _, hsz = grid.shape

        # temporal mean pooling
        grid = grid.mean(1)  # (B, H, W, d)
        grid = self.add_2d_positional_embeddings(grid)  # (B, H, W, d)
        # image token sequence
        visual_tokens = grid.view(bsz, -1, hsz)  # (B, H*W, d)

        # perform random sampling. It is only used in training phase
        # of pre-training, but not used in inference or downstream tasks.
        if hasattr(self.config, "pixel_random_sampling_size") and \
                self.config.pixel_random_sampling_size > 0 and self.training:
            sampled_indices = get_random_sample_indices(
                seq_len=visual_tokens.shape[1],
                num_samples=self.config.pixel_random_sampling_size,
                device=visual_tokens.device
            )
            visual_tokens = visual_tokens.index_select(
                dim=1, index=sampled_indices)  # (B, #samples, d)
        visual_tokens_shape = visual_tokens.shape[:-1]  # (B, H*W)
        device = visual_tokens.device

        # image token type embeddings.
        token_type_ids = torch.zeros(
            visual_tokens_shape, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = visual_tokens + position_embeddings + token_type_embeddings
        embeddings = visual_tokens + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (B, H*W, d)

    def add_temporal_postion_embeddings(self, grid):
        """
        Args:
            grid: (B, n_frms, H, W, d)

        Returns:
            (B, n_frms, H, W, d)
        """
        n_frms, height, width, hsz = grid.shape[-4:]

        # add row-wise position embeddings
        temporal_position_ids = torch.arange(
            n_frms, dtype=torch.long, device=grid.device)  # (n_frms, )
        t_position_embeddings = self.temporal_position_embeddings(
            temporal_position_ids)  # (n_frms, d)
        new_shape = (1, n_frms, 1, 1, hsz)  # (1, n_frms, 1, 1, d)
        grid = grid + t_position_embeddings.view(
            *new_shape)  # broadcast automatically

        return grid

    def add_2d_positional_embeddings(self, grid):
        """
        Args:
            grid: (B, *, H, W, d)

        Returns:
            (B, *, H, W, d)
        """
        height, width, hsz = grid.shape[-3:]

        # add row-wise position embeddings
        row_position_ids = torch.arange(
            height, dtype=torch.long, device=grid.device)  # (H, )
        row_position_embeddings = self.row_position_embeddings(
            row_position_ids)  # (H, d)
        row_shape = (1, ) * (len(grid.shape) - 3) + (
            height, 1, hsz)  # (1, *1, H, 1, d)
        grid = grid + row_position_embeddings.view(
            *row_shape)  # broadcast automatically

        # add column-wise position embeddings
        col_position_ids = torch.arange(
            width, dtype=torch.long, device=grid.device)  # (W, )
        col_position_embeddings = self.col_position_embeddings(
            col_position_ids)  # (W, d)
        col_shape = (1, ) * (len(grid.shape) - 3) + (
            1, width, hsz)  # (1, *1, 1, W, d)
        grid = grid + col_position_embeddings.view(
            *col_shape)  # broadcast automatically
        return grid


class CLIPBaseModel(nn.Module):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    config keys:
        clip_config: str, text model name, default "openai/clip-vit-base-path-32"
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.txt_model = CLIPTextModel.from_pretrained(config.pretrained_model)
        self.vis_modal = CLIPVisionModelWithProjection.from_pretrained(config.pretrained_model)

    def forward(self, txt_inputs, vis_inputs):
        r"""Modified from BertModel
        text_input_ids: (B, Lt)
        visual_inputs: (B * #frame, C, H, W)
        attention_mask: (B, Lt)  with 1 indicates valid, 0 indicates invalid position.
        """
        txt_out = self.txt_model(**txt_inputs)
        vis_out = self.vis_modal(**vis_inputs)
        return dict(txt_out=txt_out, vis_out=vis_out, txt_attn_mask=txt_inputs["attention_mask"])

class BLIPBaseModel(nn.Module):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    config keys:
        clip_config: str, text model name, default "openai/clip-vit-base-path-32"
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BlipForQuestionAnswering.from_pretrained(config.pretrained_model)

    def forward(self, inputs):
        r"""Modified from BertModel
        text_input_ids: (B, Lt)
        visual_inputs: (B * #frame, C, H, W)
        attention_mask: (B, Lt)  with 1 indicates valid, 0 indicates invalid position.
        """
        if self.training:
            outputs = self.model(**inputs) 
        else:
            outputs = self.generate(**inputs)
        return outputs.loss

def instance_bce_with_logits(logits, labels, reduction="mean"):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss

ClipBertForSequenceClassificationConfig = dict(
    cls_hidden_scale=2,   # mlp intermediate layer hidden size scaler
    classifier="mlp",  # classfied type, [mlp, linear]
    num_labels=3129,  # number of labels for classifier output
    loss_type="bce"  # [BCE, CE, KLDivLoss] only used when num_labels > 1
)

class CrossAttentionLayer(nn.Module):
    def __init__(self, in_size, dropout, nhead, n_layer=1, attn_type='dec-only', **kwargs):
        super(CrossAttentionLayer, self).__init__()
        self.attn_type = attn_type
        if attn_type == 'enc-dec':
            self.attention = torch.nn.Transformer(
                d_model=in_size,
                nhead=nhead,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dropout=dropout,
                dim_feedforward=4*in_size,
                batch_first=True,
                activation=torch.nn.functional.gelu,
            )
        elif attn_type in ['dec-only', 'dec-cas']:
            dec_layer = torch.nn.TransformerDecoderLayer(
                d_model=in_size,
                nhead=nhead,
                dim_feedforward=4*in_size,
                batch_first=True,
                activation=torch.nn.functional.relu
            )
            self.attention = torch.nn.TransformerDecoder(decoder_layer=dec_layer, num_layers=n_layer)
    
    def forward(self, txt_in, vis_in, txt_attn_mask=None):
        if self.attn_type == 'enc-dec':
            return self.attention(vis_in, txt_in, tgt_key_padding_mask=~txt_attn_mask.bool())
        elif self.attn_type == 'dec-only':
            # trg is the first param
            return self.attention(txt_in, vis_in, tgt_key_padding_mask=~txt_attn_mask)
        elif self.attn_type == 'dec-cas':
            T = vis_in.size(1)  # (B, L, E_v)
            o = txt_in
            for t in range(T):
                o = self.attention(
                    o, vis_in[:, t:t+1], 
                    tgt_key_padding_mask=~txt_attn_mask
                )
            return o
        
        
class CLIPForSeqClassification(nn.Module):
    """
    Modified from BertForSequenceClassification to support oscar training.
    """
    def __init__(self, config):
        super(CLIPForSeqClassification, self).__init__()
        self.config = config

        if 'clip' in config.pretrained_model.lower():
            self.clip = CLIPBaseModel(config)
        # elif 'blip' in config.pretrained_model.lower():
            # self.vlm = BLIPBaseModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
                
        self.attention = CrossAttentionLayer(
            in_size=config.txt_output_size, dropout=0.1, nhead=8, attn_type='dec-only'
        )
        self.classifier = nn.Linear(config.txt_output_size, config.num_labels)

    def forward(self, txt_inputs, vis_inputs, video_start_end, repeat_counts=None):
        outputs = self.clip(
            txt_inputs=txt_inputs,
            vis_inputs=vis_inputs,
        )
        txt_output, vis_output = outputs['txt_out'], outputs['vis_out']
        txt_attn_mask = outputs['txt_attn_mask']    # (B, L_t)
        vis_pooled_output = vis_output.image_embeds    # (\sum L_i, E)
        txt_pooled_output = txt_output.pooler_output    # (B, E_t)

        bsz, e_t = txt_pooled_output.size()
        decoded_tokens = txt_pooled_output.new_zeros(bsz, 1, e_t)
        txt_attn_mask = torch.cat([txt_attn_mask.new_ones(bsz, 1), txt_attn_mask], dim=1)   # (B, L_t + 1)

        # for unequal numbers of video frames
        sample_vis_outputs = []
        if repeat_counts is None:
            for s, e in zip(video_start_end[:-1],video_start_end[1:]):
                # sample_vis_outputs.append(vis_pooled_output[s:e].mean(dim=0, keepdim=True))  # List of (1, E) 
                sample_vis_outputs.append(vis_pooled_output[s:e])  # List of (L, E_v) 
            sample_vis_outputs = torch.stack(sample_vis_outputs)
        else:
            for s, e, rc in zip(video_start_end[:-1], video_start_end[1:], repeat_counts):
                sample_vis_outputs.append(vis_pooled_output[s:e].mean(dim=0).repeat(rc, 1)) # (rc, E_v)
            sample_vis_outputs = torch.cat(sample_vis_outputs, dim=0)

        txt_attn_in = torch.cat([decoded_tokens, txt_output.last_hidden_state], dim=1)
        vis_attn_in = sample_vis_outputs
        
        attn_outputs = self.attention(txt_attn_in, vis_attn_in, txt_attn_mask) # 
        logits = self.classifier(attn_outputs)[:,0,:]   # (b, V)
        return logits, None



