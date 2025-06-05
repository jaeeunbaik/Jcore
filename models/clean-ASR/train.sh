export PYTHONPATH=/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASRR:$PYTHONPATH 
python ./trainer.py --config /home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/models/clean-ASR/config_librispeech.yaml \
                    --mode train \
                    --ckpt None