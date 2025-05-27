import torch
import torch.nn as nn
import torch.nn.functional as F


class KDWrapper(nn.Module):
    def __init__(self, teacher_model, student_model, asr_loss_fn, kd_loss_fn, distil_target="encoder"):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.distil
        self.kd_loss_fn = kd_loss_fn
        self.distil_target = distil_target

    def forward(self, x, x_len, label) -> dict:
        with torch.no_grad():
            t_enc, _ = self.teacher.encoder(x, x_len)
            
        s_enc, _ = self.student.encoder(x, x_len)
        
        s_ctc_logits = self.student.ctc(s_enc, x_len)
        s_loss_ctc = self.student.ctc.loss(s_ctc_logits, x_len, label)
        
    
        total_kd_loss = 0.0
        
        if self.distil_target=='encoder' and t_enc is not None:
            enc_kd_loss = self.kd_loss_fn(t_enc, s_enc)
            kd_loss = enc_kd_loss
            total_kd_loss += enc_kd_loss
            
        if self.distil_target=='ctc' and s_ctc_logits is not None:
            t_ctc_logits = self.teacher.ctc(t_enc, x_len)
            ctc_kd_loss = self.kd_loss_fn(t_ctc_logits, s_ctc_logits)
            kd_loss = ctc_kd_loss
            total_kd_loss += ctc_kd_loss
            
        total_loss = s_loss_ctc + total_kd_loss
        
        return {
            "total_loss": total_loss,
            "kd_loss": total_kd_loss,
            "student_ctc_loss": s_loss_ctc,
        }