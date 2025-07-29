export PYTHONPATH=/home/hdd2/jenny/ASRToolkit/Jcore:$PYTHONPATH 
CUDA_VISIBLE_DEVICES=1 python ./trainer.py --config /home/hdd2/jenny/ASRToolkit/Jcore/models/kor-ASR-2/config_test.yaml \
                    --mode test \
                    --ckpt /home/hdd2/jenny/ASRToolkit/Jcore/models/kor-ASR-2/models/step=120000-val_cer=0.0000.ckpt