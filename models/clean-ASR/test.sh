export PYTHONPATH=/home/hdd2/jenny/Self-Distillation-ASR:$PYTHONPATH 
python ./trainer.py --config /home/hdd2/jenny/Self-Distillation-ASR/models/config.yaml \
                --mode test \
                --ckpt /home/hdd2/jenny/Self-Distillation-ASR/models/noise-robust-ASR/48jd5x0q/checkpoints/epoch=0-step=879.ckpt