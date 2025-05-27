import argparse
from attrdict import AttrDict

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from argparse import Namespace
import yaml
from types import SimpleNamespace
import re

from models.modelmodule import ModelModule
from models.data.datamodule import ASRDataModule
from util.utils_text import TokenProcessor

def dict_to_namespace(d):
    """딕셔너리를 네임스페이스로 변환"""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        elif isinstance(value, list):
            setattr(namespace, key, value)
        else:
            setattr(namespace, key, value)
    return namespace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='models/config.yaml')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--test_data', action='store_true', help='Debugging test_step method of ModelModule')
    parser.add_argument('--val_data', action='store_true', help='Debugging valid_step method of ModelModule')
    
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = AttrDict(config)
    # 네임스페이스로 변환
    model_config = config.model
    optim_config = config.optimizer
    tokenizer_path = config.data.tokenizer
    
    # 모델 로드 후 CPU로 이동 (약 76-77줄 부근)
    if args.ckpt:
        print(f"체크포인트에서 모델 로드: {args.ckpt}")
        model = ModelModule.load_from_checkpoint(
            args.ckpt,
            model_config=model_config, 
            optim_config=optim_config,
            tokenizer_path=tokenizer_path
        )
        # 모델을 명시적으로 CPU로 이동
        model = model.cpu()
    else:
        print("새 모델 생성")
        model = ModelModule(model_config, optim_config, tokenizer_path)

    # 모델을 평가 모드로 설정
    model.eval()
    
    # 데이터 로드
    print("데이터 모듈 초기화")
    datamodule = ASRDataModule(config)
    
    
    # 데이터 준비
    datamodule.prepare_data()
    datamodule.setup('fit')
    
    # 배치 가져오기
    if args.test_data:
        datamodule.setup('test')
        loader = datamodule.test_dataloader()
        step_method = model.test_step
    elif args.val_data:
        loader = datamodule.val_dataloader()
        step_method = model.validation_step
    else:
        loader = datamodule.train_dataloader()
        step_method = model.training_step
    # 첫 번째 배치 가져와서 디버깅
    batch = next(iter(loader))
    x, x_len, y = batch
    
    print(f"입력 형태: x={x.shape}, x_len={x_len.shape}, y={y.shape}")
    
    # 샘플 하나만 선택해서 디버깅 (메모리 절약)
    x_sample = x[:1]
    x_len_sample = x_len[:1]
    y_sample = y[:1]
    
    try:
        # 모델 메서드 직접 호출
        print(f"선택된 메서드({step_method.__name__}) 실행 중...")
        result = step_method(batch, batch_idx)
        
        print(f"메서드 실행 결과: {result}")
        
    except Exception as e:
        print(f"메서드 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    # Pure CTC 디코딩 테스트
    try:
        print("\n=== CTC 디코딩 테스트 ===")
        # 설정 객체로 변환
        decoder_config = model.model_config.asr.decoder
        
        # 디코딩 인수 설정
        decoder_config.ctc_weight = 1.0
        decoder_config.beam_size = 1
        
        # 디코딩 실행
        print("디코딩 시작...")
        decoded = model.model.recognize(x_sample, x_len_sample, y_sample, decoder_config)
        print(f"디코딩 완료: {decoded}")
        
        # 결과 확인
        if decoded and len(decoded) > 0:
            yseq = decoded[0]['yseq']
            print(f"yseq 형태: {type(yseq)}, 길이: {len(yseq)}")
            
            # TokenProcessor 초기화
            model.token_processor = TokenProcessor(model.tokenizer_path)
            id2text = model.token_processor.id2text
            
            # 텍스트 변환
            if len(yseq) > 1:  # sos 토큰 제외
                pred_tokens = yseq[1:]
                
                # 토큰 ID 출력 (디버깅용)
                print(f"토큰 ID 목록: {[t if not torch.is_tensor(t) else t.item() for t in pred_tokens[:20]]}")
                
                # 1. 각 토큰을 개별적으로 디코딩하여 리스트 생성
                token_texts = []
                for token in pred_tokens:
                    if torch.is_tensor(token):
                        token_id = token.item() if token.numel() == 1 else token.tolist()
                    else:
                        token_id = token
                        
                    # 각 토큰을 텍스트로 변환
                    token_text = id2text([token_id] if isinstance(token_id, int) else token_id)
                    token_texts.append(token_text)
                
                # 2. 토큰 목록 출력
                print(f"토큰 텍스트 목록: {token_texts[:20]}")
                
                # 3. 토큰을 공백으로 연결하고 후처리
                raw_text = " ".join(token_texts)
                print(f"Raw 디코딩 텍스트: {raw_text}")
                
                # 4. 후처리: 불필요한 공백 정리
                # - 연속된 공백을 하나로
                # - 문장 부호 앞의 공백 제거
                # - 단어 사이에 공백 유지
                cleaned_text = re.sub(r'\s+', ' ', raw_text)                  # 연속된 공백을 하나로
                cleaned_text = re.sub(r'\s([,.!?:;])', r'\1', cleaned_text)   # 문장 부호 앞 공백 제거
                cleaned_text = cleaned_text.strip()                           # 앞뒤 공백 제거
                
                # 5. SentencePiece에서 자주 발생하는 특수 처리
                # '_'로 시작하는 토큰은 SentencePiece에서 단어 중간 또는 접미사를 의미할 수 있음
                # 이런 경우 앞의 공백을 제거
                cleaned_text = re.sub(r'\s+▁', ' ', cleaned_text)  # SentencePiece 토큰 특수 처리
                cleaned_text = cleaned_text.replace('▁', ' ')      # SentencePiece 마커를 공백으로 변환
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text)   # 다시 연속된 공백 제거
                
                print(f"최종 디코딩된 텍스트: {cleaned_text}")
                
                # 6. 실제 발화와 비교
                # 원본 발화 텍스트 구하기
                gt_tokens = y_sample[0].tolist()
                gt_text = id2text(gt_tokens)
                print(f"원본 텍스트: {gt_text}")
    except Exception as e:
        print(f"디코딩 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("디버깅 완료")
    
    validate_tokenization(model, datamodule.val_dataloader().dataset, model.tokenizer_path)
    check_tokenizer_consistency(model, model.tokenizer_path)
    manual_ctc_decode(model, batch)
    check_model_output(model, batch)
    check_loss_calculation(model, batch)
    check_decoding_algorithm(model, batch)
    check_dataset_quality(datamodule)
    
    
    
    

# debug_script.py에 추가할 토큰화 검증 코드
def validate_tokenization(model, dataset, tokenizer_path):
    print("\n===== 토큰화 검증 테스트 =====")
    
    # 토크나이저 로드
    token_processor = TokenProcessor(tokenizer_path)
    
    # 데이터셋에서 몇 개의 예제 가져오기
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    examples = [next(iter(loader)) for _ in range(3)]
    
    for i, (x, x_len, y) in enumerate(examples):
        print(f"\n예제 {i+1}:")
        
        # 원본 텍스트 확인
        try:
            # Ground Truth 토큰 가져오기
            gt_tokens = y[0].tolist()
            
            # 유효한 토큰만 필터링
            valid_tokens = [t for t in gt_tokens if 0 <= t < token_processor.sp.get_piece_size() and t != -1]
            
            # 토큰 ID -> 텍스트
            gt_text = token_processor.id2text(valid_tokens)
            print(f"원본 텍스트: '{gt_text}'")
            
            # 텍스트 -> 토큰 ID -> 텍스트 (라운드트립 테스트)
            # 이 과정이 원본과 일치하는지 확인
            encoded_tokens = token_processor(gt_text)
            decoded_text = token_processor.id2text(encoded_tokens.tolist())
            print(f"재인코딩 텍스트: '{decoded_text}'")
            
            if gt_text != decoded_text:
                print("⚠️ 경고: 라운드트립 텍스트가 일치하지 않음!")
            
        except Exception as e:
            print(f"토큰화 검증 오류: {e}")

def check_model_output(model, batch):
    print("\n===== 모델 출력 형식 확인 =====")
    x, x_len, y = batch
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 인코더만 실행
    with torch.no_grad():
        # 이 부분은 모델 구조에 따라 달라질 수 있음
        if hasattr(model.model, 'encode'):
            encoder_out = model.model.encode(x, x_len)
            print(f"인코더 출력 형태: {encoder_out.shape}")
            
            # log_softmax가 적용되었는지 확인
            if hasattr(model.model, 'ctc'):
                ctc_output = model.model.ctc(encoder_out, x_len)
                
                # 출력이 log_softmax 형태인지 확인
                # log_softmax의 특성: 모든 값이 0 이하여야 함
                print(f"CTC 출력 형태: {ctc_output.shape}")
                print(f"CTC 출력 값 범위: {ctc_output.min().item()} ~ {ctc_output.max().item()}")
                
                if ctc_output.max() > 0:
                    print("⚠️ 경고: CTC 출력이 log_softmax가 아닌 것 같습니다!")
                    print("모든 값이 0 이하여야 합니다.")


def check_loss_calculation(model, batch):
    print("\n===== 손실 함수 계산 확인 =====")
    x, x_len, y = batch
    
    # y 형태 확인
    print(f"타깃 형태: y={y.shape}, y_len=추정필요")
    
    # 추정된 타깃 길이 계산 (보통 -1 값으로 패딩된 부분을 제외)
    y_lens = []
    for i in range(y.size(0)):
        length = 0
        for j in range(y.size(1)):
            if y[i, j] != -1:  # 패딩 값이 -1이라고 가정
                length += 1
        y_lens.append(length)
    
    y_len = torch.tensor(y_lens)
    print(f"추정된 타깃 길이: {y_len}")
    
    # 손실 함수 직접 계산
    try:
        with torch.no_grad():
            # 인코더 출력
            encoder_out = model.model.encode(x, x_len)
            
            # CTC 로스 계산 (직접)
            from torch.nn import CTCLoss
            ctc_loss = CTCLoss(blank=0, reduction='mean')
            
            if hasattr(model.model, 'ctc'):
                # 로짓 가져오기 (CTC 모듈이 있는 경우)
                log_probs = model.model.ctc(encoder_out, x_len)
            else:
                # 직접 로짓 계산
                log_probs = torch.log_softmax(encoder_out, dim=-1)
            
            # 패딩된 타깃 제거를 위한 마스크
            non_pad_mask = y != -1
            targets = y[non_pad_mask].view(y.size(0), -1)
            
            # CTC 손실 계산
            loss = ctc_loss(log_probs.transpose(0, 1), targets, x_len, y_len)
            print(f"수동 계산된 CTC 손실: {loss.item()}")
            
            # 모델의 공식 training_step으로 계산된 손실과 비교
            model_loss = model.training_step(batch, 0)
            if isinstance(model_loss, dict):
                model_loss = model_loss.get('loss')
            
            print(f"모델 training_step에서의 손실: {model_loss}")
            print(f"차이: {abs(loss.item() - model_loss.item()) if torch.is_tensor(model_loss) else 'N/A'}")
            
    except Exception as e:
        print(f"손실 계산 중 오류: {e}")
        import traceback
        traceback.print_exc()


def check_decoding_algorithm(model, batch):
    print("\n===== 디코딩 알고리즘 확인 =====")
    x, x_len, y = batch
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    try:
        with torch.no_grad():
            # 인코더 출력
            encoder_out = model.model.encode(x, x_len)
            
            # CTC 그리디 디코딩 (가장 간단한 방법)
            if hasattr(model.model, 'ctc'):
                log_probs = model.model.ctc(encoder_out, x_len)
            else:
                log_probs = torch.log_softmax(encoder_out, dim=-1)
            
            print(f"로그 확률 형태: {log_probs.shape}")
            
            # 가장 높은 확률의 인덱스 추출
            pred_tokens = torch.argmax(log_probs, dim=-1)
            print(f"예측 토큰 형태: {pred_tokens.shape}")
            
            # 연속된 중복 및 빈칸(0) 제거 함수
            def ctc_decode(tokens, blank_id=0):
                result = []
                prev = -1
                for t in tokens:
                    if t != blank_id and t != prev:
                        result.append(t.item())
                    prev = t
                return result
            
            # 첫 번째 샘플에 대해 디코딩
            first_sample_tokens = pred_tokens[0, :x_len[0]]
            decoded_tokens = ctc_decode(first_sample_tokens)
            
            print(f"그리디 디코딩 토큰: {decoded_tokens}")
            
            # 토큰을 텍스트로 변환
            token_processor = TokenProcessor(model.tokenizer_path)
            decoded_text = token_processor.id2text(decoded_tokens)
            print(f"디코딩된 텍스트: '{decoded_text}'")
            
            # 모델의 공식 인식 메서드와 비교
            model_decoded = model.model.recognize(x, x_len, y, model.model_config.asr.decoder)
            print(f"모델 인식 결과: {model_decoded}")
            
    except Exception as e:
        print(f"디코딩 확인 중 오류: {e}")
        import traceback
        traceback.print_exc()


def check_learning_rate(model):
    print("\n===== 학습률 확인 =====")
    
    # 옵티마이저 설정 확인
    optimizer = model.configure_optimizers()
    if isinstance(optimizer, tuple):
        optimizer = optimizer[0]  # 일부 경우 (optimizer, scheduler) 튜플 반환
    
    if isinstance(optimizer, list):
        optimizer = optimizer[0]  # 일부 경우 옵티마이저 리스트 반환
    
    # 학습률 출력
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"파라미터 그룹 {i} 학습률: {param_group['lr']}")
    
    # 학습률 스케줄러 확인
    if hasattr(model, 'lr_schedulers') and callable(model.lr_schedulers):
        print("학습률 스케줄러 구성:")
        try:
            schedulers = model.lr_schedulers()
            for i, scheduler in enumerate(schedulers):
                print(f"스케줄러 {i}: {scheduler.__class__.__name__}")
        except Exception as e:
            print(f"스케줄러 확인 중 오류: {e}")


def check_dataset_quality(datamodule):
    print("\n===== 데이터셋 품질 확인 =====")
    
    # 훈련 데이터셋 통계
    train_dataset = datamodule.train_dataloader().dataset
    val_dataset = datamodule.val_dataloader().dataset
    
    print(f"훈련 데이터 수: {len(train_dataset)}")
    print(f"검증 데이터 수: {len(val_dataset)}")
    
    # 훈련 데이터셋 샘플링
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    samples = [next(iter(train_loader)) for _ in range(5)]
    
    # 오디오 길이 분포
    x_lens = [sample[1].item() for sample in samples]
    print(f"오디오 길이 샘플: {x_lens}")
    
    # 텍스트 길이 분포 (패딩 제외)
    y_lens = [sum(sample[2][0] != -1).item() for sample in samples]
    print(f"텍스트 길이 샘플: {y_lens}")
    
    # 특이값 확인
    for i, (x, x_len, y) in enumerate(samples):
        print(f"\n샘플 {i+1}:")
        print(f"오디오 특성: shape={x.shape}, min={x.min().item():.2f}, max={x.max().item():.2f}, mean={x.mean().item():.2f}")
        
        # 텍스트 토큰 확인
        valid_tokens = y[0][y[0] != -1].tolist()
        print(f"텍스트 토큰: {valid_tokens[:20]}{'...' if len(valid_tokens) > 20 else ''}")


def check_batchnorm_status(model):
    print("\n===== 배치 노멀라이제이션 상태 확인 =====")
    
    # 모든 배치 노멀라이제이션 레이어 찾기
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
            bn_layers.append((name, module))
    
    print(f"찾은 BatchNorm 레이어 수: {len(bn_layers)}")
    
    # 일부 레이어 상태 확인
    for i, (name, bn) in enumerate(bn_layers[:5]):  # 처음 5개만 확인
        print(f"\nBatchNorm 레이어: {name}")
        print(f"- running_mean 범위: {bn.running_mean.min().item():.3f} ~ {bn.running_mean.max().item():.3f}")
        print(f"- running_var 범위: {bn.running_var.min().item():.3f} ~ {bn.running_var.max().item():.3f}")
        
        # 매우 작은 분산이나 큰 평균이 있는지 체크
        if bn.running_var.min().item() < 1e-5:
            print("⚠️ 경고: 매우 작은 분산 값 발견! 이는 훈련 문제를 나타낼 수 있습니다.")
        
    # 더 많은 레이어가 있으면 알림
    if len(bn_layers) > 5:
        print(f"\n... 추가 {len(bn_layers) - 5}개 BatchNorm 레이어 생략됨 ...")

def check_tokenizer_consistency(model, tokenizer_path):
    print("\n===== 토크나이저 일관성 확인 =====")
    
    # 토크나이저 로드
    token_processor = TokenProcessor(tokenizer_path)
    
    # 토크나이저 정보 출력
    print(f"토크나이저 경로: {tokenizer_path}")
    
    if hasattr(token_processor, 'sp'):
        vocab_size = token_processor.sp.get_piece_size()
        print(f"어휘 크기: {vocab_size}")
        
        # 모델 출력 크기 확인
        if hasattr(model.model, 'ctc'):
            output_size = model.model.ctc.ctc_lo.out_features
            print(f"모델 CTC 출력 크기: {output_size}")
            
            if vocab_size != output_size:
                print(f"⚠️ 심각한 오류: 토크나이저 어휘 크기({vocab_size})와 모델 출력 크기({output_size})가 일치하지 않습니다!")
                print("이는 매우 높은 WER의 주요 원인이 될 수 있습니다.")
            else:
                print("✓ 토크나이저 어휘 크기와 모델 출력 크기가 일치합니다.")
        
        # 몇 가지 샘플 토큰 확인
        print("\n샘플 토큰:")
        for i in range(min(vocab_size, 20)):
            piece = token_processor.sp.id_to_piece(i)
            print(f"ID {i}: '{piece}'")


def manual_ctc_decode(model, batch):
    print("\n===== 수동 CTC 디코딩 =====")
    x, x_len, y = batch
    
    with torch.no_grad():
        # 인코더 출력 확인
        encoder_out = model.model.encode(x, x_len)
        print(f"인코더 출력 형태: {encoder_out.shape}")
        
        # CTC 출력 확인
        ctc_output = model.model.ctc(encoder_out, x_len, y)
        print(f"CTC 출력 형태: {ctc_output.shape}")
        
        # 최대 확률 토큰 확인
        pred_tokens = torch.argmax(ctc_output, dim=-1)
        print(f"예측 토큰 형태: {pred_tokens.shape}")
        
        # 첫 번째 샘플만 처리
        sample_pred = pred_tokens[0].cpu().numpy()
        
        # 연속된 중복 및 빈칸(blank) 제거
        decoded_seq = []
        prev_token = -1
        for token in sample_pred:
            if token != 0 and token != prev_token:  # blank=0 가정
                decoded_seq.append(token)
            prev_token = token
            
        print(f"디코딩된 시퀀스: {decoded_seq[:20]}...")
        
        # 토큰을 텍스트로 변환
        token_processor = TokenProcessor(model.tokenizer_path)
        decoded_text = token_processor.id2text(decoded_seq)
        print(f"수동 디코딩 텍스트: '{decoded_text}'")
        
        # 모델 인식 결과와 비교
        model_decoded = model.model.recognize(x, x_len, y, model.model_config.asr.decoder)
        model_text = "인식 실패"
        if model_decoded and len(model_decoded) > 0:
            yseq = model_decoded[0]['yseq']
            if len(yseq) > 1:  # SOS token 제외
                model_tokens = yseq[1:]
                model_tokens = [t.item() if torch.is_tensor(t) else t for t in model_tokens]
                model_text = token_processor.id2text(model_tokens)
                
        print(f"모델 인식 텍스트: '{model_text}'")


if __name__ == "__main__":
    main()