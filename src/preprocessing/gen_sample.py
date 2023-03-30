import argparse, os
import h5py, json
from tqdm import tqdm
import torch
from torch import nn
import random
import numpy as np
from datautils.utils import Timer
from src.utils.basic_utils import load_jsonl, load_json, save_json, get_rounded_percentage
# from datautils import tgif_qa
from datautils import msrvtt_qa
from datautils import msvd_qa
# from datautils import svqa
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter


def get_cap(processor, cap_model, frms):
    bsz = frms.shape[0]
    input_ids = processor(text=['[CLS] ']*bsz, add_special_tokens=False, return_tensors='pt').input_ids.cuda()
    pixel_values = torch.Tensor(frms.reshape(bsz, 3, 224, 224)).cuda()
    generated_captions = processor.batch_decode(cap_model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=30), skip_special_tokens=True, max_length=30)
    return generated_captions

def generate_cap(processor, caption_model, anno_path, h5_outfile, vid_map_file, args):
    vid2id = load_json(vid_map_file)
    caption_model.eval().cuda()
    caption_model = torch.nn.DataParallel(caption_model, device_ids=[0, 1, 2, 3])
    if isinstance(caption_model, torch.nn.DataParallel):
        caption_model = caption_model.module
    
    with h5py.File(h5_outfile, 'r') as fd:
        sampled_frames = fd['sampled_frames']
        for split in ['train', 'val', 'test']:
            anno_file = os.path.join(anno_path, 'qa_{}.json'.format(split))
            ds = load_json(anno_file)
            new_ds = []
            for sample in tqdm(ds[:2]):
                new_sample = sample.copy()
                vid = sample['video'].split('.')[0]
                idx = vid2id[vid]
                
                # read frms
                frms = sampled_frames[idx]
                caps = get_cap(processor, caption_model, frms)    
                new_sample['caps'] = caps
                
                new_ds.append(new_sample)

            new_anno_file = os.path.join(anno_path, 'qa_new_{}.json'.format(split))
            save_json(new_ds, new_anno_file)

def generate_inds(tokenizer, model, anno_path, args):
    for split in ['train', 'val', 'test']:
        anno_file = os.path.join(anno_path, 'qa_{}.json'.format(split))
        ds = load_json(anno_file)
        new_ds = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset info
    parser.add_argument('--dataset', default='msvd_qa', choices=['msvd_qa', 'msrvtt_qa', 'svqa'], type=str)
    parser.add_argument('--dataset_root', default='./dataset', type=str)
    # output
    parser.add_argument('--out', dest='outfile', help='output filepath', default="{}_video_feat.h5", type=str)

    # feature extraction hps
    parser.add_argument('--chunk_size', type=int, default=512, help='chunk size for computing feature similarity')
    parser.add_argument('--img_size', type=int, default=224, help='image size of extracted frames')
    parser.add_argument('--intv', type=int, default=1, help='sampling interval between video frames')
    parser.add_argument('--K', type=int, default=32, help='number of frames to be sampled (esp. uniform sampling)')

    # network params
    parser.add_argument('--vlm_model', type=str, default="microsoft/git-base-coco")
    parser.add_argument('--h5_path', type=str, default="processed")
    parser.add_argument('--task', type=str, choices=['gen_cap', 'gen_inds'], default='gen_cap')
    args = parser.parse_args()
    args.seed = 666

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = os.path.join(args.dataset_root, args.dataset)

    if args.task == 'gen_cap':
    # initialize caption model and processor (default GIT)
        processor = AutoProcessor.from_pretrained(args.vlm_model)
        if 'git' in args.vlm_model.lower():
            caption_model = AutoModelForCausalLM.from_pretrained(args.vlm_model)
            LOGGER.info('git pretrained model loaded')
        else:
            raise ValueError('No such captioning model implementations')

        # annotation files
        if args.dataset == 'msrvtt_qa':
            pass
        elif args.dataset == 'msvd_qa':
            # annotation file, h5 file, mapping file
            dataset_path = os.path.join(args.dataset_root, args.dataset)

            # files under h5 path
            h5_path = os.path.join(dataset_path, args.h5_path)
            h5_outfile = os.path.join(h5_path, args.outfile.format(args.dataset))
            vid_map_file = os.path.join(h5_path, 'vidmapping.json')
            
            anno_path = os.path.join(dataset_path, 'annotations')
            # generate captions
            generate_cap(processor, caption_model, anno_path, h5_outfile, vid_map_file, args)

        elif args.dataset == 'svqa':
            args.annotation_file = './data/SVQA/questions.json'
            args.video_dir = './data/SVQA/useVideo/'
            video_paths = svqa.load_video_paths(args)
            random.shuffle(video_paths)
            # load model
            generate_cap(processor, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.feature_type, str(args.num_clips)))
            
    elif args.task == 'gen_idx':    # text-only models
        tokenizer = AutoTokenizer.from_pretrained(args.vlm_model)
        model = AutoModelForSequenceClassification.from_pretrained(args.vlm_model)
        
        if args.dataset == 'msrvtt_qa':
            pass
        elif args.dataset == 'msvd_qa':
            # annotation file, h5 file, mapping file
            dataset_path = os.path.join(args.dataset_root, args.dataset)
            anno_path = os.path.join(dataset_path, 'annotations')

            # generate captions
            generate_inds(tokenizer, model, anno_path, args)

