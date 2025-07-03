export PYTHONPATH=/home/hdd2/jenny/ASRToolkit/Jcore:$PYTHONPATH 
CUDA_LAUNCH_BLOCKING=1 python ./trainer.py --config /home/hdd2/jenny/ASRToolkit/Jcore/models/clean-ASR/config_librispeech.yaml \
                    --mode train \
                    --ckpt None