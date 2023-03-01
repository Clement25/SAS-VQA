from torch import nn
from src.datasets.data_utils import repeat_tensor_rows
from src.utils.load_save import load_state_dict_with_mismatch
from transformers import CLIPVisionModel, CLIPTextModel
from torch.nn import CrossEntropyLoss, MSELoss
from .modeling import instance_bce_with_logits
from .modeling import CLIPForSeqClassification

# model wrapper
class CLIPModelforFinetune(nn.Module):
    def __init__(self, config, vlm_cls=CLIPForSeqClassification):
        super(CLIPModelforFinetune, self).__init__()
        self.config = config
        self.VLModel = vlm_cls(config)
        
    def forward(self, batch):
        # used to make visual feature copies
        repeat_counts = batch["n_examples_list"]
        del batch["n_examples_list"]
        visual_features = self.cnn(batch["visual_inputs"])
        batch["visual_inputs"] = repeat_tensor_rows(
            visual_features, repeat_counts)
        if self.retrieval:
            batch["sample_size"] = len(repeat_counts)  # batch size
        
        # obtain outputs
        logits = self.VLModel(
                                txt_inputs=batch['txt_inputs'],
                                vis_inputs=batch['vis_inputs']
                            )
        
        logits, loss = self.calc_loss(logits, batch['labels'])
        return dict(logits=logits, loss=loss)

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False

    def calc_loss(self, logits, labels):
        if labels is not None:
            if self.config.num_labels == 1:  # regression
                loss_fct = MSELoss(reduction="none")
                # labels = labels.to(torch.float)
                loss = loss_fct(
                    logits.view(-1), labels.view(-1))
            else:
                if self.config.loss_type == 'bce':  # [VQA]
                    loss = instance_bce_with_logits(
                        logits, labels, reduction="none")
                elif self.config.loss_type == "ce":  # cross_entropy [GQA, Retrieval, Captioning]
                    loss_fct = CrossEntropyLoss(reduction="none")
                    loss = loss_fct(
                        logits.view(-1, self.config.num_labels),
                        labels.view(-1))
                else:
                    raise ValueError("Invalid option for config.loss_type")
        else:
            loss = 0
        return logits, loss