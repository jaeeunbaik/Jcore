export PYTHONPATH=/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR:$PYTHONPATH 
CUDA_VISIBLE_DEVICES=3 python ./trainer.py --config /home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/models/kor-ASR/config_test.yaml \
                    --mode test \
                    --ckpt /home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/models/kor-ASR/models/epoch=04-val_wer=0.1441.ckpt