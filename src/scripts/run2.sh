mode=$1

if [[ $mode = 'train' ]];then
    CUDA_VISIBLE_DEVICES=2 python tasks/run_video_qa.py --task msvd_qa --config ./configs/msvd_qa_base2.json --ans2label_path ../dataset/msvd_qa/annotations/train_ans2label.json 
else
    CUDA_VISIBLE_DEVICES=2 python tasks/run_video_qa.py --task msvd_qa --config ./configs/msvd_qa_base2.json --ans2label_path ../dataset/msvd_qa/annotations/train_ans2label.json --do_inference 1
fi