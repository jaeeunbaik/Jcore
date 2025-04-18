export PYTHONPATH=/home/hdd2/jenny/Self-Distillation-ASR:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1 python ./trainer.py --config /home/hdd2/jenny/Self-Distillation-ASR/models/config.yaml \
        --mode train