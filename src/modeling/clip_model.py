from torch import nn
from src.datasets.data_utils import repeat_tensor_rows
from src.utils.load_save import load_state_dict_with_mismatch
from transformers import CLIPVisionModel, CLIPTextModel

class CLIPModelforFinetune(nn.Module):
    def __init__(self, config, freeze=False):
        super(CLIPModelforFinetune, self).__init__()
        self.config = config

        self.txt_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch-32")
        self.vis_modal = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch-32")

    def forward(self, batch):
        # used to make visual feature copies
        repeat_counts = batch["n_examples_list"]
        del batch["n_examples_list"]
        visual_features = self.cnn(batch["visual_inputs"])
        batch["visual_inputs"] = repeat_tensor_rows(
            visual_features, repeat_counts)
        if self.retrieval:
            batch["sample_size"] = len(repeat_counts)  # batch size
        outputs = self.transformer(**batch)  # dict
        return outputs

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False