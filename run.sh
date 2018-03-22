# export PATH="/usr/local/nvidia/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
# source /tools/config.sh
# source activate py27

CUDA_VISIBLE_DEVICES="1"  python train_11.py --checkpoint _baseline --model 11
CUDA_VISIBLE_DEVICES="1"  python train_10.py --checkpoint 11 --model 10
CUDA_VISIBLE_DEVICES="1"  python train_9.py --checkpoint 10 --model 9
CUDA_VISIBLE_DEVICES="1"  python train_8.py --checkpoint 9 --model 8
CUDA_VISIBLE_DEVICES="1"  python train_7.py --checkpoint 8 --model 7
CUDA_VISIBLE_DEVICES="1"  python train_6.py --checkpoint 7 --model 6
CUDA_VISIBLE_DEVICES="1"  python train_5.py --checkpoint 6 --model 5
CUDA_VISIBLE_DEVICES="1"  python train_4.py --checkpoint 5 --model 4 
CUDA_VISIBLE_DEVICES="1"  python train_3.py --checkpoint 4 --model 3
CUDA_VISIBLE_DEVICES="1"  python train_2.py --checkpoint 3 --model 2
CUDA_VISIBLE_DEVICES="1"  python train_1.py --checkpoint 2 --model 1
CUDA_VISIBLE_DEVICES="1"  python finetune.py --checkpoint 1 --model finetuned