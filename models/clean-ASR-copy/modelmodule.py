"""
🖤🐰 JaeEun Baik, 2025
"""
import re
import random 

import jiwer
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from typing import List

import sentencepiece as spm

from util.utils_text import TokenProcessor, ErrorCalculator, preprocess_text
from modules.loss.kd_loss import KDLoss
from modules.kd_wrapper import KDWrapper
from modules.e2e_asr_model import e2eASR



class ModelModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model_config = config.model.asr
        self.kd_config = config.model.distillation
        self.use_kd = self.kd_config.using_distillation
        # self.kd_loss = KDLoss(model_config.distillation)
        # if self.use_kd:
        #     self.teacher_model = e2eASR(model_config.teacher)
        #     self.student_model = e2eASR(model_config.student)
            
        #     self.model = KDWrapper(self.teacher_model, self.student_model, self.kd_loss, model_config.distillation.target)
        # else:
        self.optim_config = config.optimizer
        self.lr = np.float64(self.optim_config.op_lr)
        self.tokenizer_path = config.data.tokenizer
        self.token_processor = TokenProcessor(config.data.tokenizer)
        self.trainer_config = config.trainer
        self.model = e2eASR(self.model_config, self.tokenizer_path)
        
        self.sp_processor = spm.SentencePieceProcessor()
        self.sp_processor.load(self.tokenizer_path)
        
        self.char_list = [self.sp_processor.id_to_piece(i) for i in range(self.model_config.decoder.odim)]
        self.error_calculator = ErrorCalculator(
            char_list=self.char_list,
            sym_space=self.model_config.sym_space,
            sym_blank=self.model_config.sym_blank,
            report_cer=self.model_config.report_cer,
            report_wer=self.model_config.report_wer
        )
        
    def log_gradient_norms(self):
        total_norm = 0.0
        module_norms = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                module = name.split('.')[0]
                norm = param.grad.norm().item()
                if module not in module_norms:
                    module_norms[module] = []
                module_norms[module].append(norm)
                total_norm += norm ** 2
        total_norm = total_norm ** 0.5
        
        # 로그 기록
        self.log("grad/total_norm", total_norm, prog_bar=False)
        for module, norms in module_norms.items():
            avg_norm = sum(norms) / len(norms)
            self.log(f"grad/{module}_norm", avg_norm, prog_bar=False)
    
    def print_model_structure(self):
        """모델 구조 및 파라미터 출력"""
        print("\n===== 모델 구조 =====")
        total_params = 0
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 말단 모듈만 출력
                params = sum(p.numel() for p in module.parameters())
                total_params += params
                print(f"{name}: {module.__class__.__name__}, 파라미터 수: {params:,}")
        
        print(f"총 파라미터 수: {total_params:,}")
        
        # encoder와 ctc 모듈 확인
        print("\n===== 주요 모듈 확인 =====")
        if hasattr(self.model, 'encoder'):
            print(f"Encoder type: {type(self.model.encoder)}")
        else:
            print("Encoder not found!")
            
        if hasattr(self.model, 'ctc'):
            print(f"CTC type: {type(self.model.ctc)}")
        else:
            print("CTC module not found!")    

    # 모델 구조랑 파라미터랑 파라미터수 출력할려면
    def on_train_start(self):
        """훈련 시작 시 호출되는 메서드"""
        super().on_train_start()
        # self.print_model_structure()
        # 첫 번째 배치에 대해 미사용 파라미터 확인
        # print("\n==== 미사용 파라미터 확인 시작 ====")
    #     self._parameter_debugging_done = False

    def on_train_batch_end(self, batch_output, batch, batch_idx):
        """각 배치 시작 시 호출되는 메서드"""
        # 첫 배치에서만 디버깅 실행
        if batch_idx == 0 and not getattr(self, '_parameter_debugging_done', False):
            x, x_len, y = batch
            # 배치 크기가 너무 크면 첫 번째 샘플만 사용
            if x.size(0) > 1:
                x = x[:1]
                x_len = x_len[:1]
                if y is not None and not isinstance(y, dict):
                    y = y[:1] if len(y.shape) > 0 else y
            
            # print("\n디버깅 샘플 형태:")
            # print(f"x: {x.shape}, x_len: {x_len.shape}")
            if y is not None:
                if isinstance(y, dict):
                    y_shape = {k: v.shape for k, v in y.items()}
                else:
                    y_shape = y.shape
                # print(f"y: {y_shape}")
            
            # 파라미터 사용 상태 확인
            # _, _ = self.check_unused_parameters(x, x_len, y)
            # self._parameter_debugging_done = True
            
            # print("\n==== 미사용 파라미터 확인 완료 ====\n")
    
    def training_step(self, batch, batch_idx):
        x, x_len, y = batch
        if self.use_kd:
            loss_dict = self.model(x, x_len, y)
            self.log("train/total_loss", loss_dict["total_loss"])
            self.log("train/kd_loss", loss_dict["kd_loss"])
            self.log("train/asr_loss", loss_dict["student_loss"])
            return loss_dict["total_loss"]
        else:
            loss = self.model(x, x_len, y)
            log_items = {
                "train/total_loss": loss.get("loss"),
                "train/ctc_loss": loss.get("loss_ctc"),
                "train/cer": loss.get("cer"),
                "train/wer": loss.get("wer"),
            }
            
            # None이 아닌 값만 log로 넘김
            for key, value in log_items.items():
                if value is not None:
                    self.log(key, value, prog_bar=True, sync_dist=True)
            ctc_probs = self.model.calculate_all_ctc_probs(x, x_len, y)
            if ctc_probs is not None:
                confidence = torch.tensor(ctc_probs).max(-1)[0].mean().item()
                self.log("train/ctc_confidence", confidence, prog_bar=True)

            # --- Attention visualization ---
            attn_weights = self.model.calculate_all_attentions(x, x_len, y)
            if "decoder.0.self_attn" in attn_weights:
                attn_matrix = attn_weights["decoder.0.self_attn"][0]
                self.logger.experiment.add_image(
                    "train/attention", torch.tensor(attn_matrix).mean(0, keepdim=True), self.global_step
                )

                self.log_gradient_norms()
            return loss.get("loss")


    def validation_step(self, batch, batch_idx):
        x, x_len, y = batch
        if self.use_kd:
            loss_dict = self.model(x, x_len, y)
            self.log("val/loss", loss_dict["total_loss"])
            self.log("val/kd_loss", loss_dict["kd_loss"])
            self.log("val/asr_loss", loss_dict["student_loss"])
        else:
            loss_output = self.model(x, x_len, y) # `loss` 변수명을 `loss_output`으로 변경하여 혼동 방지
            log_items = {
                "val/loss": loss_output.get("loss"),
                "val/ctc_loss": loss_output.get("loss_ctc"),
                "val/att_loss": loss_output.get("loss_att"),
            }

            # None이 아닌 값만 log로 넘김
            for key, value in log_items.items():
                if value is not None:
                    self.log(key, value, prog_bar=True, sync_dist=True)
            
            # CTC Confidence 로깅 (필요하다면)
            ctc_probs = self.model.calculate_all_ctc_probs(x, x_len, y)
            if ctc_probs is not None:
                confidence = torch.tensor(ctc_probs).max(-1)[0].mean().item()
                self.log("val/ctc_confidence", confidence, prog_bar=True)

            # Attention visualization (필요하다면)
            attn_weights = self.model.calculate_all_attentions(x, x_len, y)
            if "decoder.0.self_attn" in attn_weights:
                attn_matrix = attn_weights["decoder.0.self_attn"][0]
                self.logger.experiment.add_image(
                    "val/attention", torch.tensor(attn_matrix).mean(0, keepdim=True), self.global_step
                )
            # --- 모델 디코딩 및 WER/CER 계산 (새로운 로직) ---
            with torch.no_grad():
                # self.model_config.decoder는 Namespace 객체로, 필요한 recog_args를 포함해야 함.
                # (예: beam_size, maxlenratio, minlenratio, lm_weight 등)
                recog_args = self.model_config.decoder 
                decoded_raw_output = self.model.recognize(x, x_len, y, recog_args=recog_args)
            
            # 예측된 텍스트 리스트를 담을 변수
            predicted_transcriptions: List[str] = []
            # 디코더 타입에 따라 decoded_raw_output 처리
            if self.model.decoder_type in ['ctc', 'rnnt']:
                # ctc와 rnnt 디코딩은 recognize에서 이미 List[str]을 반환하도록 수정했다고 가정
                if isinstance(decoded_raw_output, list) and all(isinstance(d, str) for d in decoded_raw_output):
                    predicted_transcriptions = decoded_raw_output
                else:
                    if isinstance(decoded_raw_output, dict) and decoded_raw_output:
                        first_key = next(iter(decoded_raw_output))
                        if isinstance(decoded_raw_output[first_key], list) and all(isinstance(s, str) for s in decoded_raw_output[first_key]):
                            predicted_transcriptions = decoded_raw_output[first_key]
                        else:
                            self.log_text("val_debug/decode_type_mismatch", "RNNT/CTC output is not List[str] as expected.", self.global_step)
                            predicted_transcriptions = [""] * len(x) # 오류 방지를 위해 빈 문자열로 채움
                    else:
                        self.log_text("val_debug/decode_type_mismatch", "RNNT/CTC output is not List[str] or Dict as expected.", self.global_step)
                        predicted_transcriptions = [""] * len(x) # 오류 방지를 위해 빈 문자열로 채움
            else: # 그 외 지원하지 않는 디코더 타입
                self.log_text("val_debug/unsupported_decoder_type", f"Unsupported decoder type: {self.model.decoder_type}", self.global_step)
                predicted_transcriptions = [""] * len(x) # 오류 방지를 위해 빈 문자열로 채움

            # Ground Truth 텍스트 변환 (모든 디코더 타입에 공통)
            reference_transcriptions: List[str] = []
            for i in range(len(y)):
                gt_tokens = y[i].tolist()
                gt_text = self.token_processor.id2text(gt_tokens, filter_blank=True) # TokenProcessor가 잘 처리하는지 중요!
                reference_transcriptions.append(gt_text)
            # 텍스트 전처리 (WER 계산을 위해 공통적으로 수행)
            # preprocess_text 함수가 util/utils_text에 있다고 가정
            processed_predicted_transcriptions = [preprocess_text(text) for text in predicted_transcriptions]
            processed_reference_transcriptions = [preprocess_text(text) for text in reference_transcriptions]

            batch_wers = []
            for gt, pred in zip(processed_reference_transcriptions, processed_predicted_transcriptions):
                if gt and pred: # 빈 문자열이 아닐 때만 WER 계산
                    wer = jiwer.wer(gt, pred)
                    batch_wers.append(wer)
                else:
                    # 빈 텍스트로 인해 WER 계산이 불가능한 경우 (로그 출력)
                    self.log_text("val_debug/empty_text_for_wer", f"Empty text for WER calculation: GT='{gt}', PR='{pred}'", self.global_step)
            # 평균 WER 계산 및 로깅
            if batch_wers:
                avg_wer = sum(batch_wers) / len(batch_wers)
                self.log("val_wer", avg_wer, prog_bar=True, sync_dist=True)
            else:
                self.log("val_wer", 1.0, prog_bar=True, sync_dist=True) # 모든 WER 계산 불가능 시 1.0 (최악) 로깅

            # 각 배치의 첫 번째 샘플에 대한 출력 (디버깅용)
            if batch_idx == 0 and len(processed_reference_transcriptions) > 0:
                print(f"\n===== 배치 {batch_idx}, 샘플 0 (Decoder Type: {self.model.decoder_type}) =====")
                print(f"GT: '{processed_reference_transcriptions[0]}'")
                print(f"PR: '{processed_predicted_transcriptions[0]}'")
                if batch_wers:
                    print(f"WER: {batch_wers[0]:.4f} ({batch_wers[0] * 100:.2f}%)")
                else:
                    print("WER 계산 불가 (빈 텍스트)")
                    
                    
            # loss = self.model(x, x_len, y)
            # log_items = {
            #     "val/loss": loss.get("loss"),
            #     "val/ctc_loss": loss.get("loss_ctc"),
            #     "val/att_loss": loss.get("loss_att"),
            #     "val/wer": loss.get("wer"),
            #     "val/cer": loss.get("cer"),
            # }

            # # None이 아닌 값만 log로 넘김
            # for key, value in log_items.items():
            #     if value is not None:
            #         self.log(key, value, prog_bar=True, sync_dist=True)
            # ctc_probs = self.model.calculate_all_ctc_probs(x, x_len, y)
            # if ctc_probs is not None:
            #     confidence = torch.tensor(ctc_probs).max(-1)[0].mean().item()
            #     self.log("val/ctc_confidence", confidence, prog_bar=True)

            # # --- Attention visualization ---
            # attn_weights = self.model.calculate_all_attentions(x, x_len, y)
            # if "decoder.0.self_attn" in attn_weights:
            #     attn_matrix = attn_weights["decoder.0.self_attn"][0]
            #     self.logger.experiment.add_image(
            #         "val/attention", torch.tensor(attn_matrix).mean(0, keepdim=True), self.global_step
            #     )


            # # 모델 디코딩
            # with torch.no_grad():
            #     decoded = self.model.recognize(x, x_len, y, recog_args=self.model_config.decoder)

            # batch_wers = []

            # if decoded:  # decoded = [{'score': score, 'yseq': [2, token1, ...]}, ... ]
            #     for i in range(len(decoded)):  # 배치 크기만큼 반복
            #         # print(f"[DEBUG] decoded 정보 len(decoded[0]['yseq']) : {len(decoded[0]['yseq'])}")
            #         # print(f"[DEBUG] decoded 정보 len(decoded[0]['yseq'][1:]) : {len(decoded[0]['yseq'][1:])}")
            #         # print(f"[DEBUG] decoded 정보 decoded[0]['yseq'][1:][i] : {decoded[0]['yseq'][1:][i]}")
            #         if 'yseq' in decoded[i]:
            #             yseq = decoded[i]['yseq'][1:]  # SOS 토큰 제외
            #             pred_text = self.token_processor.id2text(yseq, filter_blank=True) if yseq else ""

            #             # Ground Truth 텍스트 변환
            #             gt_tokens = y[i].tolist() if i < len(y) else []
            #             gt_text = self.token_processor.id2text(gt_tokens, filter_blank=True)

            #             # 텍스트 전처리
            #             pred_text = preprocess_text(pred_text)
            #             gt_text = preprocess_text(gt_text)

            #             # WER 계산
            #             if gt_text and pred_text:
            #                 wer = jiwer.wer(gt_text, pred_text)
            #                 batch_wers.append(wer)

            #                 # 각 배치의 첫 번째 샘플에 대해 예측 및 실제 텍스트 출력
            #                 if i == 0:  # 첫 번째 샘플
            #                     print(f"\n===== 배치 {batch_idx}, 샘플 {i} =====")
            #                     print(f"GT: '{gt_text}'")
            #                     print(f"PR: '{pred_text}'")
            #                     print(f"WER: {wer:.4f} ({wer * 100:.2f}%)")
            #             else:
            #                 if i == 0:  # 첫 번째 샘플
            #                     print(f"\n===== 배치 {batch_idx}, 샘플 {i} =====")
            #                     print("빈 텍스트가 있어 WER를 계산할 수 없습니다.")

            #     # 평균 WER 계산 및 로깅
            #     if batch_wers:
            #         avg_wer = sum(batch_wers) / len(batch_wers)
            #         self.log("val_wer", avg_wer, prog_bar=True)
            #     else:
            #         self.log("val_wer", 0, prog_bar=True)
                                
                
                
    def on_validation_epoch_start(self):
        """에폭 시작 시 WER 통계 초기화"""
        self.val_wer_samples = []
        self.val_wer_sum = 0
        self.val_wer_count = 0

    def on_validation_epoch_end(self):
        """에폭 종료 시 WER 통계 처리 및 로깅"""
        if self.val_wer_count > 0:
            avg_wer = self.val_wer_sum / self.val_wer_count
            self.log("val_wer", avg_wer)
            
            # 히스토그램 로깅
            if self.logger and hasattr(self.logger, "experiment"):
                import numpy as np
                import wandb
                if self.val_wer_samples:
                    self.logger.experiment.log({
                        "val_wer_histogram": wandb.Histogram(np.array(self.val_wer_samples)),
                        "global_step": self.global_step
                    })
            
            print(f"\n===== 검증 완료 =====")
            print(f"검증 샘플 수: {self.val_wer_count}")
            print(f"평균 WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")    
                
                
    def test_step(self, batch, batch_idx):
        x, x_len, y = batch
        if self.use_kd:
            logits = self.student_model.encode(x, x_len)
            decoded_trans = self.student_model.recognize(logits)
        else:
            decoded_trans = self.model.recognize(x, x_len, y, self.model_config.decoder)

        ref_trans: List[str] = []
        for single_y_tokens in y:
            # ignore_id (padding)를 제외하고 실제 토큰만 사용
            # self.model.sos, self.model.eos 도 고려하여 제거해야 할 수 있습니다.
            # 이 부분은 토크나이저 및 데이터셋 구성에 따라 달라질 수 있습니다.
            filtered_tokens = [
                token.item() for token in single_y_tokens 
                if token.item() != self.model.ignore_id and 
                   token.item() != self.model.sos and 
                   token.item() != self.model.eos
            ]
            ref_trans.append(self.token_processor.id2text(filtered_tokens))
        
        cer_batch = self.error_calculator.calculate_cer(decoded_trans, ref_trans)
        wer_batch = self.error_calculator.calculate_wer(decoded_trans, ref_trans)
        
        # 평균 CER/WER 로깅 (배치 단위)
        self.log("test_cer", cer_batch, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_wer", wer_batch, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # 필요하다면 추가적인 로깅 또는 결과 반환
        return {
            "decoded_transcriptions": decoded_trans,
            "reference_transcriptions": ref_trans,
            "cer_batch": cer_batch,
            "wer_batch": wer_batch
        }
        
        # # 텍스트로 변환
        # self.token_processor = TokenProcessor(self.tokenizer_path)
        # id2text = self.token_processor.id2text
        
        # # 결과 확인
        # if decoded and len(decoded) > 0:
        #     yseq = decoded[0]['yseq']
            
        #     # 텍스트 변환 (SOS 토큰 제외)
        #     if len(yseq) > 1:
        #         pred_tokens = yseq[1:]
                
        #         # 각 토큰을 개별적으로 디코딩하여 리스트 생성
        #         token_texts = []
        #         for token in pred_tokens:
        #             if torch.is_tensor(token):
        #                 token_id = token.item() if token.numel() == 1 else token.tolist()
        #             else:
        #                 token_id = token
                        
        #             # 각 토큰을 텍스트로 변환
        #             token_text = id2text([token_id] if isinstance(token_id, int) else token_id)
        #             token_texts.append(token_text)
                
        #         # 토큰을 공백으로 연결하고 후처리
        #         raw_text = " ".join(token_texts)
                
        #         # 후처리: 불필요한 공백 정리
        #         cleaned_text = re.sub(r'\s+', ' ', raw_text)                  # 연속된 공백을 하나로
        #         cleaned_text = re.sub(r'\s([,.!?:;])', r'\1', cleaned_text)   # 문장 부호 앞 공백 제거
        #         cleaned_text = cleaned_text.strip()                           # 앞뒤 공백 제거
                
        #         # SentencePiece에서 자주 발생하는 특수 처리
        #         cleaned_text = re.sub(r'\s+▁', ' ', cleaned_text)  # SentencePiece 토큰 특수 처리
        #         cleaned_text = cleaned_text.replace('▁', ' ')      # SentencePiece 마커를 공백으로 변환
        #         cleaned_text = re.sub(r'\s+', ' ', cleaned_text)   # 다시 연속된 공백 제거
                
        #         # GT 토큰 필터링 및 텍스트 변환
        #         try:
        #             # Ground Truth 텍스트 처리
        #             gt_tokens_raw = y[0].tolist() if y.dim() > 1 else y.tolist()
                    
        #             # -1 및 특수 토큰 필터링 (0: padding, 1: sos, 2: eos, -1: ignore_id)
        #             valid_tokens = []
        #             for t in gt_tokens_raw:
        #                 # 유효한 토큰 범위 확인 (토크나이저 사전 크기에 따라 조정 필요)
        #                 max_token_id = self.token_processor.sp.get_piece_size() - 1
                        
        #                 # 유효한 범위의 토큰만 포함
        #                 if 0 <= t <= max_token_id:
        #                     valid_tokens.append(t)
        #                 elif t == -1:
        #                     # -1 토큰(ignore_id)은 출력하지 않음
        #                     continue
        #                 else:
        #                     # 예상치 못한 토큰 ID 디버깅
        #                     self.print(f"[WARNING] 범위를 벗어난 토큰 ID: {t}")
                    
        #             # 필터링된 토큰으로 텍스트 변환
        #             if valid_tokens:
        #                 gt_text = id2text(valid_tokens)
        #             else:
        #                 gt_text = "[토큰 없음]"
                    
        #         except Exception as e:
        #             self.print(f"[ERROR] GT 텍스트 변환 중 오류 발생: {e}")
        #             gt_text = "[처리 오류]"
                
        #         # 예시 출력
        #         self.print(f"\n--- 테스트 샘플 {batch_idx} ---")
        #         self.print(f"GT: {gt_text}")
        #         self.print(f"PR: {cleaned_text}")
                
        #         # WER 계산 (jiwer 라이브러리 사용)
        #         try:
        #             import jiwer
                    
        #             # 텍스트 전처리
        #             gt_processed = preprocess_text(gt_text)
        #             pr_processed = preprocess_text(cleaned_text)
                    
        #             # 전처리된 텍스트가 빈 문자열이면 계산 생략
        #             if len(gt_processed.strip()) > 0 and len(pr_processed.strip()) > 0:
        #                 # WER 계산
        #                 sample_wer = jiwer.wer(gt_processed, pr_processed)
                        
        #                 # WER을 로그로 저장
        #                 self.log(f"test_sample_wer", sample_wer, on_step=True)
                        
        #                 # 통계를 추적하기 위한 전역 변수 업데이트
        #                 if not hasattr(self, 'wer_sum'):
        #                     self.wer_sum = 0.0
        #                     self.wer_count = 0
                        
        #                 self.wer_sum += sample_wer
        #                 self.wer_count += 1
                        
        #                 # 현재 평균 WER 계산 및 로깅
        #                 current_avg_wer = self.wer_sum / self.wer_count
        #                 self.log("test_avg_wer", current_avg_wer, on_step=True)
                        
        #                 # WER 출력
        #                 self.print(f"WER: {sample_wer:.4f} ({sample_wer*100:.2f}%)")
        #                 self.print(f"현재까지 평균 WER: {current_avg_wer:.4f} ({current_avg_wer*100:.2f}%)")
        #                 self.print(f"처리된 샘플 수: {self.wer_count}")
        #             else:
        #                 self.print("텍스트가 비어 있어 WER 계산을 건너뜁니다.")
                    
        #         except ImportError:
        #             self.print("jiwer 라이브러리가 없어 WER 계산을 건너뜁니다. 'pip install jiwer'로 설치하세요.")
        #         except Exception as e:
        #             self.print(f"WER 계산 중 오류: {str(e)}")
                
        #         return {"text": cleaned_text, "wer": current_avg_wer}
        
        # # 디코딩 결과가 없거나 오류가 발생한 경우
        # self.print(f"\n--- 테스트 샘플 {batch_idx} ---")
        # self.print("디코딩 실패")
        # return {"text": "디코딩 실패"}

            
    def configure_optimizers(self):
        # optimizer
        if self.optim_config.type == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
        elif self.optim_config.type == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            
        # scheduler
        def transformer_lr_schedule(step, d_model, warmup_steps):
            """
            Transformer Learning Rate Schedule.

            Args:
                step (int): 현재 스텝.
                d_model (int): 모델의 차원 (hidden size).
                warmup_steps (int): 워밍업 스텝 수.

            Returns:
                float: 학습률 스케일링 값.
            """
            if step == 0:
                step = 1  # 0으로 나누는 것을 방지
            scale = d_model ** -0.5
            return scale * min(step ** -0.5, step * (warmup_steps ** -1.5))
        d_model = self.model_config.encoder.encoder_dim
        warmup_steps = self.optim_config.warmup_steps
        def lr_lambda(step):
            return transformer_lr_schedule(step, d_model, warmup_steps)

        if self.optim_config.scheduling_type == "cosine-annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer_config.num_epochs)  # T_max : cosine 주기 한 번 도는데 걸리는
        elif self.optim_config.scheduling_type == "warmup":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)
        elif self.optim_config.scheduling_type == "lambda":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=2**10,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
            
            
    def check_unused_parameters(self, x, x_len, y):
        """
        모델의 사용 및 미사용 파라미터를 확인하는 함수
        """
        # 그래디언트 초기화
        self.model.train()
        self.model.zero_grad()
        
        # 파라미터 이름과 requires_grad 상태 저장
        param_status_before = {}
        for name, param in self.model.named_parameters():
            param_status_before[name] = {
                'requires_grad': param.requires_grad,
                'grad': param.grad,
            }
        
        # 포워드 및 백워드 패스 수행
        loss = self.model(x, x_len, y)
        if isinstance(loss, dict):
            loss_value = loss.get('loss')
            if loss_value is not None:
                loss_value.backward()
        else:
            loss.backward()
        
        # 사용/미사용 파라미터 확인
        used_params = []
        unused_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    unused_params.append(name)
                else:
                    # 그래디언트가 0이 아닌 요소가 있는지 확인
                    if param.grad.abs().sum().item() > 0:
                        used_params.append(name)
                    else:
                        unused_params.append(name)
        
        print(f"\n===== 총 파라미터 수: {len(param_status_before)} =====")
        print(f"사용된 파라미터 수: {len(used_params)} ({len(used_params)/len(param_status_before):.2%})")
        print(f"미사용 파라미터 수: {len(unused_params)} ({len(unused_params)/len(param_status_before):.2%})")
        
        # 미사용 파라미터 출력 (선택적으로 사용)
        if unused_params:
            print("\n미사용 파라미터 목록:")
            for name in unused_params:
                print(f"- {name}")
        
        # 모델 구조별 사용/미사용 파라미터 비율 분석
        module_stats = {}
        for name in param_status_before.keys():
            # 모듈 이름 추출 (첫 번째 dot까지)
            module_name = name.split('.')[0] if '.' in name else 'base'
            
            if module_name not in module_stats:
                module_stats[module_name] = {'used': 0, 'unused': 0, 'total': 0}
            
            module_stats[module_name]['total'] += 1
            if name in used_params:
                module_stats[module_name]['used'] += 1
            else:
                module_stats[module_name]['unused'] += 1
        
        print("\n모듈별 파라미터 사용 현황:")
        for module_name, stats in module_stats.items():
            used_percent = stats['used'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"{module_name}: {stats['used']}/{stats['total']} 사용 ({used_percent:.1f}%)")
        
        # 그래디언트 초기화
        self.model.zero_grad()
        
        return used_params, unused_params
    
    
    def log_text(self, key: str, value: str, step: int):
        """
        WandB에 텍스트를 로깅하기 위한 헬퍼 함수.
        PyTorch Lightning의 logger.experiment를 통해 WandB API에 접근합니다.
        """
        if self.logger and hasattr(self.logger, "experiment") and isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
            # self.logger.experiment는 WandB Run 객체입니다.
            # wandb.log()를 사용하여 텍스트를 로깅합니다.
            self.logger.experiment.log({key: value}, step=step)
        else:
            # WandB 로거가 활성화되지 않았거나 다른 로거를 사용하는 경우 콘솔에 출력
            print(f"Log (Step {step}) - {key}: {value}")