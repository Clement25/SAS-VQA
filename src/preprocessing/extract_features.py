import argparse, os
import h5py, json
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
import torchvision
import random
import numpy as np
from datautils.utils import sample_representative_frames, sample_frames_uniform, Timer
# from datautils import tgif_qa
# from datautils import msrvtt_qa
from datautils import msvd_qa
# from datautils import svqa
from transformers import CLIPImageProcessor, CLIPVisionModel
from prefetch_loader import *
from queue import Queue, Empty, Full
from threading import Thread
from collections import Counter

def generate_vidid_json(video_paths, json_outfile):
    mapping_dict = {}
    for i, video_path in enumerate(video_paths):
        video_id = video_path.split('/')[-1].split('.')[0]
        mapping_dict[video_id] = i
    json.dump(mapping_dict, open(json_outfile, 'w'))

def generate_h5(processor, model, video_paths, args, h5_outfile):  # default-8
    """
    Args:
        model: loaded pretrained model for feature extraction
        video_ids: list of video ids
        num_clips: expected numbers of splitted clips
        outfile: path of output file to be written
    Returns:
        h5 file containing visual features of splitted clips.
    """
    if not os.path.exists('data/{}'.format(args.dataset)):
        os.makedirs('data/{}'.format(args.dataset))

    dataset_size = len(video_paths)  # paths
    # sampling hps and criterions
    num_frames_default = getattr(args, 'nfrms', 32)
    
    with h5py.File(h5_outfile, 'w') as fd:
        feat_dset = None
        flens = None
        i0 = 0
        _t = {'misc': Timer()}
        for i, video_path in enumerate(tqdm(video_paths)):
            _t['misc'].tic()
            frames, valid = extract_clips_with_consecutive_frames(video_path, processor, model, args)
            video_length = len(frames)
            if args.feature_type == 'video':
                frames_torch = torch.FloatTensor(np.asarray(frames['pixel_values'])).cpu()
                if valid:
                    frames_torch = frames_torch.squeeze()
                else:
                    frames_torch = np.zeros(shape=(num_frames_default, 768)) # clip output hidden dimension is 768
                if feat_dset is None:
                    c, h, w = frames_torch.shape[-3:]
                    feat_dset = fd['processed_feats'] = np.zeros(shape=(len(video_paths), 128, c, h, w))
                    flens = fd['frame_lengths'] = np.zeros(shape=(len(video_paths), 1))

            i1 = i0 + 1
            # if exceeds the maximum lengths, cut the frames
            flens[i0] = video_length
            feat_dset[i0][:video_length] = frames_torch
            i0 = i1
            _t['misc'].toc()
            if (i % 1000 == 0):
                print('{:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                      .format(i1, dataset_size, _t['misc'].average_time,
                              _t['misc'].average_time * (dataset_size - i1) / 3600))


def generate_h5_parallel(processor, model, video_paths, args, h5_outfile):
    if not os.path.exists('data/{}'.format(args.dataset)):
        os.makedirs('data/{}'.format(args.dataset))
    
    # move model to cuda, set it to eval mode
    model.eval()
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    
    # cpu video queue
    memory_video_queue = Queue(maxsize=8)
    # cuda processing queue
    cuda_video_queue = Queue(maxsize=2)
    
    # video frame generator
    frm_generator = InputGen(video_paths, processor, args.intv)
    load_thread_killer = thread_killer()
    load_thread_killer.set_tokill(False)
    preprocess_workers = 1
    
    # launch 4 threads to do load && pre-process the input video frames
    for _ in range(preprocess_workers):
        t = Thread(target=threaded_batches_feeder, args=(load_thread_killer, memory_video_queue, frm_generator))
        t.start()
    
    cuda_transfers_thread_killer = thread_killer()
    cuda_transfers_thread_killer.set_tokill(False)

    cudathread = Thread(target=threaded_cuda_batches, \
                args=(cuda_transfers_thread_killer, cuda_video_queue, memory_video_queue))
    cudathread.start()
    # let queue get filled
    time.sleep(8)

    debug_counter = {'Failure': 0}
    with h5py.File(h5_outfile, 'w') as fd:    
        if args.sampling_strategy == 'uni':
            fd.create_dataset("sampled_frames", (len(video_paths), args.K, 3*224*224))
            sampled_frames_h5 = fd["sampled_frames"]
        for i in range(len(video_paths)):
            # read video frames out of the queue
            _, video_frms = cuda_video_queue.get(block=True)
            
            # extract special representative frames
            if args.sampling_strategy == 'repr':
                # FIXME: remove the counter
                exted_frms = sample_representative_frames(video_frms, model, args.K, args.W, debug_counter)
            elif args.sampling_strategy == 'uni':
                exted_frms = sample_frames_uniform(video_frms, K=args.K)
                frms_to_store = exted_frms.reshape(args.K, -1).cpu()
                sampled_frames_h5[i] = frms_to_store
    
    load_thread_killer.set_tokill(True)
    cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(preprocess_workers):
        try:
            # Enforcing thread to shut down
            memory_video_queue.get(block=True, timeout=1)
            cuda_video_queue.get(block=True, timeout=1)
        except Empty:
            pass
    
    # FIXME: remove this
    print('Total Failure:%d'%debug_counter['Failure'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # dataset info-选定数据集
    parser.add_argument('--dataset', default='msvd_qa', choices=['msvd_qa', 'msrvtt-qa', 'svqa'], type=str)
    parser.add_argument('--dataset_root', default='./dataset', type=str)
    parser.add_argument('--question_type', default='none', choices=['none'], type=str)
    # output
    parser.add_argument('--out', dest='outfile', help='output filepath', default="{}_{}_feat.h5", type=str)

    # feature extraction hps
    parser.add_argument('--chunk_size', type=int, default=512, help='chunk size for computing feature similarity')
    parser.add_argument('--intv', type=int, default=1, help='sampling interval between video frames')
    parser.add_argument('--sampling_strategy', default='uni', choices=['uni', 'repr'], type=str)
    parser.add_argument('--K', type=int, default=16, help='number of frames to be sampled (esp. uniform sampling)')
    parser.add_argument('--W', type=int, default=8, help='interval length to sample 2 points')

    # network params
    parser.add_argument('--vlm_model', type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument('--h5_fname', type=str, default="processed")
    args = parser.parse_args()
    args.seed = 666
    args.feature_type = 'video'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # initialize clip processors
    processor = CLIPImageProcessor.from_pretrained(args.vlm_model)
    vision_model = CLIPVisionModel.from_pretrained(args.vlm_model)
    dataset_path = os.path.join(args.dataset_root, args.dataset)

    # annotation files
    if args.dataset == 'tgif-qa':
        args.annotation_file = './data/tgif-qa/csv/Total_{}_question.csv'
        args.video_dir = './data/tgif-qa/gifs'
        video_paths = tgif_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        generate_h5(processor, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.feature_type, str(args.num_clips)))

    if args.dataset == 'msrvtt-qa':
        args.annotation_file = './data/MSRVTT-QA/{}_qa.json'
        args.video_dir = './data/MSRVTT-QA/video/'
        video_paths = msrvtt_qa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        generate_h5(processor, video_paths, args.num_clips,
                    args.outfile.format(args.dataset, args.feature_type, str(args.num_clips)))

    if args.dataset == 'msvd_qa':
        args.annotation_file = os.path.join(dataset_path, 'annotations/qa_{}.json')
        args.video_dir = os.path.join(dataset_path, 'video')
        video_paths = msvd_qa.load_video_paths(args)
        random.shuffle(video_paths)
        
        outpath = os.path.join(dataset_path, args.h5_fname)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        h5_outfile = os.path.join(outpath, args.outfile.format(args.dataset, args.feature_type))
        json_outfile = os.path.join(outpath, 'vidmapping.json')
        
        dataset = msvd_qa
        # generate mapping dict
        if not os.path.exists(outpath):
            generate_vidid_json(video_paths, outpath)
        # generate h5 file
        generate_h5_parallel(processor, vision_model, video_paths, args,
                    h5_outfile)

    elif args.dataset == 'svqa':
        args.annotation_file = './data/SVQA/questions.json'
        args.video_dir = './data/SVQA/useVideo/'
        video_paths = svqa.load_video_paths(args)
        random.shuffle(video_paths)
        # load model
        generate_h5(processor, video_paths, args.num_clips,
                args.outfile.format(args.dataset, args.feature_type, str(args.num_clips)))

