"""
🖤🐰 JaeEun Baik
"""
import torch
import numpy as np
import pytorch_lightning as pl

from util.utils_text import TokenProcessor
from modules.loss.kd_loss import KDLoss
from modules.kd_wrapper import KDWrapper
from modules.e2e_asr_model import e2eASR



class ModelModule(pl.LightningModule):
    def __init__(self, model_config, optim_config, tokenizer_path):
        super().__init__()
        self.model_config = model_config
        self.kd_config = model_config.distillation
        self.use_kd = self.kd_config.using_distillation
        self.kd_loss = KDLoss(temperature=2.0, alpha=1.0)
        if self.use_kd:
            self.teacher_model = e2eASR(model_config.teacher)
            self.student_model = e2eASR(model_config.student)
            
            self.model = KDWrapper(self.teacher_model, self.student_model, self.kd_loss)
        else:
            self.model = e2eASR(model_config.asr)
        self.optim_config = optim_config
        self.lr = np.float64(self.optim_config.op_lr)
        self.tokenizer_path = tokenizer_path
    
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
                "train/att_loss": loss.get("loss_att"),
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

            # --- Prediction output logging ---
            # decoded = self.model.recognize(x[0].cpu().numpy(), recog_args=self.model_config.asr.decoder)
            # if decoded:
            #     id2text = self.token_processor.id2text
            #     pred_tokens = decoded[0]['yseq'][1:]
            #     pred_text = "".join([id2text(tokens.tolist()) for tokens in pred_tokens])
            #     gt_text = "".join([id2text(t.tolist()) for t in y])
                # self.logger.log_text(
                #     key="val/sample_pred",
                #     columns=["GT", "PR"],
                #     data=[[gt_text, pred_text]],
                #     step=self.global_step
                # )
            return loss.get("loss")

    def validation_step(self, batch, batch_idx):
        x, x_len, y = batch

        if self.use_kd:
            loss_dict = self.model(x, x_len, y)
            self.log("val/total_loss", loss_dict["total_loss"])
            self.log("val/kd_loss", loss_dict["kd_loss"])
            self.log("val/asr_loss", loss_dict["student_loss"])
        else:
            loss = self.model(x, x_len, y)
            log_items = {
                "val/total_loss": loss.get("loss"),
                "val/ctc_loss": loss.get("loss_ctc"),
                "val/att_loss": loss.get("loss_att"),
                "val/cer": loss.get("cer"),
                "val/wer": loss.get("wer"),
            }

            # None이 아닌 값만 log로 넘김
            for key, value in log_items.items():
                if value is not None:
                    self.log(key, value, prog_bar=True, sync_dist=True)
            ctc_probs = self.model.calculate_all_ctc_probs(x, x_len, y)
            if ctc_probs is not None:
                confidence = torch.tensor(ctc_probs).max(-1)[0].mean().item()
                self.log("val/ctc_confidence", confidence, prog_bar=True)

            # --- Attention visualization ---
            attn_weights = self.model.calculate_all_attentions(x, x_len, y)
            if "decoder.0.self_attn" in attn_weights:
                attn_matrix = attn_weights["decoder.0.self_attn"][0]
                self.logger.experiment.add_image(
                    "val/attention", torch.tensor(attn_matrix).mean(0, keepdim=True), self.global_step
                )

            # --- Prediction output logging ---
            self.token_processor = TokenProcessor(self.tokenizer_path)
            # decoded = self.model.recognize(x[0], x_len, y, recog_args=self.model_config.asr.decoder)
            # if decoded:
            #     id2text = self.token_processor.id2text
            #     pred_tokens = decoded[0]['yseq'][1:]
            #     pred_text = "".join([id2text(tokens.tolist()) for tokens in pred_tokens])
            #     gt_text = "".join([id2text(t.tolist()) for t in y])
            #     self.logger.log_text(
            #         key="val/sample_pred",
            #         columns=["GT", "PR"],
            #         data=[[gt_text, pred_text]],
            #         step=self.global_step
            #     )
            
            
    def test_step(self, batch, batch_idx):
        x, x_len, y = batch

        if self.use_kd:
            logits = self.student_model.encode(x, x_len)
            pred_tokens = self.student_model.decode_from_logits(logits)
        else:
            logits = self.model.encode(x, x_len)
            pred_tokens = self.model.decode_from_logits(logits)

        # 텍스트로 변환
        id2text = self.token_processor.id2text  # ← TokenProcessor 인스턴스 접근
        pred_texts = [id2text(tokens.tolist()) for tokens in pred_tokens]

        # 원본 라벨도 텍스트로 변환 (원할 경우)
        gt_texts = [id2text(t.tolist()) for t in y]

        # 예시 출력
        for i in range(len(pred_texts)):
            self.print(f"GT: {gt_texts[i]}")
            self.print(f"PR: {pred_texts[i]}")

        return pred_texts

            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.optim_config.max_epochs)  # T_max : cosine 주기 한 번 도는데 걸리는
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
            
            
    