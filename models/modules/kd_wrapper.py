import torch
import torch.nn as nn
import torch.nn.functional as F


class KDWrapper(nn.Module):
    def __init__(self, teacher_model, student_model, asr_loss_fn, kd_loss_fn, distil_final_output=True):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.asr_loss_fn = asr_loss_fn
        self.kd_loss_fn = kd_loss_fn
        self.distil_final_output = distil_final_output

    def forward(self, x, label) -> dict:
        with torch.no_grad():
            t_enc, t_dec = self.teacher(x)
            
        s_enc, s_dec = self.student(x)
        s_loss, loss_ctc, loss_att = self.asr_loss_fn(s_enc, s_dec)
        
        total_loss, kd_loss = self.kd_loss_fn(t_logits, s_logits, s_loss)
        
        return {
            "total_loss": total_loss,
            "kd_loss": kd_loss,
            "student_loss": s_loss,
            "student_ctc_loss": s_loss_ctc,
            "student_attn_loss": s_loss_attn
        }