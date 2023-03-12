import torch
import random
import numpy as np
import copy
from torch.utils.data.dataloader import default_collate
from src.utils.basic_utils import flat_list_of_lists
from src.utils.load_save import LOGGER
from src.datasets.dataset_base import BaseDataset
# for debug
from transformers import CLIPTokenizerFast
from src.datasets.data_utils import mk_input_group
from collections import defaultdict

IGNORE_INDEX=-100

class VideoQADataset(BaseDataset):
    """ This should work for both train and test (where labels are not available).
    task_type: str, one of [action, frameqa, transition]
        where action and transition are multiple-choice QA,
            frameqa is opened QA similar to VQA.
    datalist: list(tuples)  each tuple is (img_id, list(dicts)),
        each dict
    tokenizer:
    max_img_size: int,
    max_txt_len: int, max text sequence length, including special tokens.
    itm_neg_prob: float [0, 1] set to 0 will disable itm.
    return_label: bool, whether return label in __getitem__
    random_sample_clips:
    """

    def __init__(self, task_type, datalist, tokenizer, img_hdf5_dir,
                 fps=3, num_frm=3, max_img_size=1000, max_txt_len=20, ans2label=None, vid2id=None, ensemble_n_clips=1, return_label=True, is_train=True, random_sample_clips=True):
        super(VideoQADataset, self).__init__(
            datalist, tokenizer, img_hdf5_dir,
            fps=fps, num_frm=num_frm,
            max_img_size=max_img_size, max_txt_len=max_txt_len)
        self.open_ended_qa_names = ["frameqa", "msrvtt_qa", "msvd_qa"]
        self.ensemble_n_clips = ensemble_n_clips
        self.return_label = return_label
        self.is_train = is_train
        self.task_type = task_type
        self.ans2label = ans2label
        self.num_labels = len(ans2label)
        self.vid2id = vid2id
        self.random_sample_clips = random_sample_clips
        self.label2ans = {v: k for k, v in ans2label.items()}
        self.qid2data = {d["question_id"]: d for group in datalist for d in group[1]}

    def __len__(self):
        return len(self.datalist)

    def _load_video_frames(self, vid):
        idx = self.vid2id[vid]
        all_frames = self.dataset[idx]
        return all_frames

    def __getitem__(self, index):
        # skip error videos:
        num_retries = 3
        for _ in range(num_retries):
            vid, examples = self.datalist[index]  # one video with multiple examples
            vid_frm_array = self._load_video_frames(vid)
            # vid_frm_array = torch.zeros_like(vid_frm_array)
            # Select a random video if the current video was not able to access.
            if vid_frm_array is None:
                LOGGER.info(f"Failed to load examples with video: {vid}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            examples = [self._get_single_example(e) for e in examples]
            return dict(
                vid=vid_frm_array,
                examples=examples,
                n_examples=len(examples)  # used to create image feature copies.
            )
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

    def _get_single_example(self, data):
        example = dict(
            q_str=data["question"],
            question_id=data["question_id"],
            label=data["answer"]
        )
        if self.task_type in ["action", "transition"]:
            example["options_str_list"] = data["options"]
        elif self.task_type in self.open_ended_qa_names:
            if self.return_label:
                example["str_label"] = example["label"] # for tokenization use
                example["label"] = self.ans2label.get(example["label"], IGNORE_INDEX)
        if not self.return_label:
            example["label"] = None
        return example

    def evaluate_tgif_qa(self, results):
        """
        Args:
            results: list(dict),
              each dict is
                {
                    "question_id": int,
                    "answer": int or float, either answer_idx (int)
                }
        Returns:
            TGIF-QA score
        """
        preds, gts = [], []
        # for frameQA
        answer_types = []
        answer_type2idx = dict(
            frameqa={"object": 0, "number": 1, "color": 2, "location": 3},
            msrvtt_qa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])},
            msvd_qa={k: idx for idx, k in enumerate(["what", "who", "how", "where", "when"])},
        )

        # qid: wid
        qid2pred_ans = {r["question_id"]: r["answer"] for r in results}
            
        for qid, pred_ans in qid2pred_ans.items():
            if type(pred_ans) is list:
                preds.extend(pred_ans)
            else:
                preds.append(pred_ans)
            gt_data = self.qid2data[qid]
            gt_ans = self.ans2label.get(gt_data["answer"], IGNORE_INDEX)
            if self.task_type in self.open_ended_qa_names:
                answer_types.append(answer_type2idx[self.task_type][gt_data["answer_type"]])
            gts.append(gt_ans)

        preds, gts = np.array(preds), np.array(gts)
        metrics = dict()
        # preds and gts are array of strings
        # ignore special indices
        if IGNORE_INDEX in gts:
            metrics["overall_acc"] = float(sum(preds == gts) / sum(gts != IGNORE_INDEX))
        else:
            metrics["overall_acc"] = float(np.mean(preds == gts))

        if self.task_type in self.open_ended_qa_names:
            answer_types = np.array(answer_types)
            ratios = dict()
            for ans_type, ans_type_idx in answer_type2idx[self.task_type].items():
                answer_type_mask = answer_types == ans_type_idx
                answer_type_corrects = (
                        preds[answer_type_mask] == gts[answer_type_mask])
                metrics[f"{ans_type}_acc"] = float(
                    np.mean(answer_type_corrects)) if len(answer_type_corrects) != 0 else 0
                ratios[f"{ans_type}_ratio"] = [
                    1. * len(answer_type_corrects) / len(answer_types),
                    len(answer_type_corrects)]
            metrics["ratios"] = ratios
        return metrics

class BaseQACollator(object):
    def __init__(self, max_length=20, task_type="action", n_options=5, nframe=4, samp_policy='random', img_size=224):
        self.max_length = max_length
        self.task_type = task_type
        self.n_options = n_options
        self.nframe = nframe
        self.samp_policy = samp_policy
        self.img_size = img_size
    
    def collate_batch(self, batch):
        raise NotImplementedError("collate function hasn't been implemented")

class VideoQACollator(BaseQACollator):
    def __init__(self, tokenizer, max_length=20, task_type="action", n_options=5, nframe=4, samp_policy='random', img_size=224):
        super(VideoQACollator, self).__init__(
            max_length=max_length, task_type=task_type,
            n_options=n_options, nframe=nframe, samp_policy=samp_policy, img_size=img_size
        )
        self.tokenizer = tokenizer

    def collate_batch(self, batch):
        v_collate = default_collate
        visual_inputs = v_collate([d["vid"] for d in batch])  # (B, T, 3, H, W)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.
        if self.task_type in ["action", "transition"]:
            text_str_list = flat_list_of_lists(
                [[d["q_str"] + " " + d["options_str_list"][i] for i in range(self.n_options)]
                 for d in text_examples]
            )  # (B * n_options, )
        else:
            text_str_list = [d["q_str"] for d in text_examples]  # (B, )
            
        batch_enc = self.tokenizer(text_str_list, padding = True, truncation = True, return_tensors='pt')

        text_input_ids = batch_enc.input_ids  # (B, L)
        text_attention_mask = batch_enc.attention_mask  # (B, L)
        
        bsz, orig_l, _ = visual_inputs.size()
        if self.samp_policy == 'uniform':
            T = orig_l // self.nframe + (1 if orig_l % self.nframe > 0 else 0)
            inds = [int(i*self.nframe) for i in range(T)]
            visual_inputs = visual_inputs[:,inds]
        elif self.samp_policy == 'random':
            rand_sample = torch.arange(orig_l).float().expand(bsz, -1)
            inds = torch.multinomial(rand_sample, num_samples=self.nframe, replacement=False)
            vinds = torch.arange(bsz).unsqueeze(-1).expand(bsz, inds.size(-1))
            visual_inputs = visual_inputs[vinds,inds]
        else:
            raise ValueError("Sample strategy can only be chosen from ['uniform', 'random']")
        B, L, _ = visual_inputs.size()
        # assert L == self.nframe
        visual_inputs = visual_inputs.reshape(B*L, 3, self.img_size, self.img_size)
        video_lengths = [L] * B
        
        video_start_end = [0]
        for l in video_lengths:
            video_start_end.append(video_start_end[-1] + l)

        labels = default_collate([int(d["label"]) for d in text_examples]) \
            if text_examples[0]["label"] is not None else None  # (B, #ans)
        question_ids = [d["question_id"] for d in text_examples]
        
        return dict(
            visual_inputs=visual_inputs,  # (B * #frm, C, H, W)
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            question_ids=question_ids,
            video_start_end=video_start_end,
            labels=labels,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )

class BLIPVideoQACollator(BaseQACollator):
    def __init__(self, processor, max_length=20, task_type="action", n_options=5, nframe=4, samp_policy='random', img_size=384):
        super(BLIPVideoQACollator, self).__init__(
            max_length=max_length, task_type=task_type,
            n_options=n_options, nframe=nframe, samp_policy=samp_policy, img_size=img_size
        )
        self.processor = processor

    def collate_batch(self, batch):
        v_collate = default_collate
        visual_inputs = v_collate([d["vid"] for d in batch])  # (B, T, 3, H, W)
        # group data
        text_examples = flat_list_of_lists([d["examples"] for d in batch])
        n_examples_list = [d["n_examples"] for d in batch]  # (B, )
        # group elements data
        # directly concatenate question and option as a single seq.
        if self.task_type in ["action", "transition"]:
            text_str_list = flat_list_of_lists(
                [[d["q_str"] + " " + d["options_str_list"][i] for i in range(self.n_options)]
                 for d in text_examples]
            )  # (B * n_options, )
        else:
            text_str_list = [d["q_str"] for d in text_examples]  # (B, )
            
        bsz, orig_l, _ = visual_inputs.size()
        if self.samp_policy == 'uniform':
            T = orig_l // self.nframe + (1 if orig_l % self.nframe > 0 else 0)
            inds = [int(i*self.nframe) for i in range(T)]
            visual_inputs = visual_inputs[:,inds]
        elif self.samp_policy == 'random':
            rand_sample = torch.arange(orig_l).float().expand(bsz, -1)
            inds = torch.multinomial(rand_sample, num_samples=self.nframe, replacement=False)
            vinds = torch.arange(bsz).unsqueeze(-1).expand(bsz, inds.size(-1))
            visual_inputs = visual_inputs[vinds,inds]
        elif self.samp_policy == 'single':
            i = orig_l//2
            visual_inputs = visual_inputs[:, i:i+1]
        else:
            raise ValueError("Sample strategy can only be chosen from ['uniform', 'random']")
        
        # FIXME: only impl single here
        batch_enc = self.processor(text=text_str_list, padding = True, 
                                   truncation = True, return_tensors='pt')
        text_input_ids = batch_enc.input_ids  # (B, L)
        text_attention_mask = batch_enc.attention_mask  # (B, L)
        

        B, L, _ = visual_inputs.size()
        # assert L == self.nframe
        visual_inputs = visual_inputs.reshape(B*L, 3, self.img_size, self.img_size)
        video_lengths = [L] * B
        
        video_start_end = [0]
        for l in video_lengths:
            video_start_end.append(video_start_end[-1] + l)

        # BLIP labels
        labels = self.processor(text=[d['str_label'] for d in text_examples], padding='longest', return_tensors='pt') if 'str_label' in text_examples[0] else None
        # else:
        #     labels = default_collate([int(d["label"]) for d in text_examples]) \
        #         if text_examples[0]["label"] is not None else None  # (B, #ans)
        question_ids = [d["question_id"] for d in text_examples]
        
        return dict(
            visual_inputs=visual_inputs,  # (B * #frm, C, H, W)
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            question_ids=question_ids,
            video_start_end=video_start_end,
            labels=labels.input_ids if labels is not None else None,
            n_examples_list=n_examples_list  # used to create image feature copies.
        )