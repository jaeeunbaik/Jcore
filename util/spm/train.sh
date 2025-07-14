#!/bin/bash
# filepath: /home/hdd2/jenny/Self-Distillation-ASR/util/spm/train.sh

nbpe=3000
bpemode=bpe
mkdir -p ${bpemode}
dict=${bpemode}/kor_${bpemode}${nbpe}_units.txt
bpemodel=${bpemode}/kor_${bpemode}${nbpe}

# 1단계: SentencePiece 모델 학습
python spm_train.py \
  --input=input.txt \
  --model_prefix=${bpemodel} \
  --vocab_size=${nbpe} \
  --model_type=${bpemode} \
  --character_coverage=1.0 \
  --input_sentence_size=100000000 \
  --pad_id=0 \
  --unk_id=1 \
  --bos_id=2 \
  --eos_id=3 \
  --control_symbols="<blank>"

# 2단계: 사전 파일 생성 - 특수 토큰을 명시적으로 추가
echo "<blank> 0" > ${dict}
echo "<unk> 1" >> ${dict}
echo "<s> 2" >> ${dict}
echo "</s> 3" >> ${dict}

# 3단계: SentencePiece 모델에서 어휘 추출 (인덱스 조정)
# 특수 토큰을 제외한 나머지 토큰을 인덱스 4부터 추가
python -c "
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('${bpemodel}.model')
with open('${dict}', 'a') as f:
  for i in range(4, sp.get_piece_size()):
    piece = sp.id_to_piece(i)
    f.write(f'{piece} {i}\n')
"

echo "토크나이저 학습 및 사전 생성 완료: ${bpemodel}.model, ${dict}"