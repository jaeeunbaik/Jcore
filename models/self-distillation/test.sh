export PYTHONPATH=/home/hdd2/jenny/ASRToolkit/Jcore:$PYTHONPATH 
python ./trainer.py --config/home/hdd2/jenny/ASRToolkit/Jcore/models/self-distillation/config_test.yaml \
                    --mode test \
                    --ckpt /home/hdd2/jenny/ASRToolkit/Jcore/models/self-distillation/models/epoch=00-val_cer=0.436.ckpt