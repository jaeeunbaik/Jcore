export PYTHONPATH=/home/hdd2/jenny/Self-Distillation-ASR:$PYTHONPATH 
# CUDA_VISIBLE_DEVICES=0,1,2,3 
python ./trainer.py --config ./config.yaml \
        --mode train