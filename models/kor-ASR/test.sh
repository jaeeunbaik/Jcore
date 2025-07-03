export PYTHONPATH=/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR:$PYTHONPATH 
python ./trainer.py --config /home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/models/kor-ASR/config_test.yaml \
                    --mode test \
                    --ckpt /home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/models/kor-ASR/models/epoch=17-val_wer=0.0000.ckpt