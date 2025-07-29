export PYTHONPATH=/home/hdd2/jenny/ASRToolkit/Jcore:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
python3 ./trainer.py --config /home/hdd2/jenny/ASRToolkit/Jcore/models/kor-ASR-2/config_kspon_AIHub.yaml \
                    --mode train