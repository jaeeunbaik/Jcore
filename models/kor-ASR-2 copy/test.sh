export PYTHONPATH=/home/hdd2/jenny/ASRToolkit/Jcore:$PYTHONPATH 
python ./trainer.py --config/home/hdd2/jenny/ASRToolkit/Jcore/models/kor-ASR-2/config_test.yaml \
                    --mode test \
                    --ckpt /home/hdd2/jenny/ASRToolkit/Jcore/models/kor-ASR-2/models/epoch=00-val_cer=0.436.ckpt