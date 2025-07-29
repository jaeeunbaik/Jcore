import logging

import numpy as np
import torch
import torch.nn.functional as F
from packaging.version import parse as V
from util.utils_module import to_device



class CTC(torch.nn.Module):
    """CTC module

    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout_rate (0.0 ~ 1.0)
    :param str ctc_type: builtin
    :param bool reduce: reduce the CTC loss into a scalar
    """
    def __init__(self, ctc_config):
        super().__init__()
        self.odim = ctc_config.odim
        self.eprojs = ctc_config.eprojs
        self.dropout_rate = ctc_config.dropout_rate
        self.ctc_type = ctc_config.ctc_type
        self.reduce = ctc_config.reduce
        
        self.dropout_rate = float(self.dropout_rate)
        self.dropout_rate = self.dropout_rate
        self.loss = None
        self.ctc_lo = torch.nn.Linear(self.eprojs, self.odim)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.probs = None
        
            
        reduction_type = "sum" if self.reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(
            reduction=reduction_type, zero_infinity=True
        )
        
        self.ignore_id = -1
        self.reduce = self.reduce
        
    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        try:
            th_pred = th_pred.log_softmax(2)
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            size = th_pred.size(1)

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss
        except:
            raise NotImplementedError
        
    def forward(self, hs_pad, hlens, ys_pad):  # hs_pad: encoder output, ys_pad: targets
        """CTC forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        """

        ys_lens_list = []
        for y_seq in ys_pad:
            # ignore_id가 아닌 실제 레이블만 카운트
            ys_lens_list.append((y_seq != self.ignore_id).sum().item())
        ys_lens = torch.tensor(ys_lens_list, dtype=torch.long, device=hlens.device)
        
        invalid_mask = hlens < (2 * ys_lens - 1)
        if invalid_mask.any():
            logging.warning("❗ CTC 조건 위반 샘플 있음")
            logging.warning(f"     - hlens:   {hlens[invalid_mask].tolist()}")
            logging.warning(f"     - ys_lens: {ys_lens[invalid_mask].tolist()}")
            # CTC 조건 위반 샘플에 대한 처리가 필요할 수 있습니다.
            # zero_infinity=True 설정 시 이런 경우 loss가 0이 됩니다.

        ys_hat = self.ctc_lo(self.dropout(hs_pad)).permute(1, 0, 2) # (Tmax, B, C)
        ys_hat = torch.clamp(ys_hat, min=-1e8, max=1e8) 

        targets_1d = torch.cat([y_seq[y_seq != self.ignore_id] for y_seq in ys_pad])

        loss = self.loss_fn(ys_hat, targets_1d, hlens, ys_lens) # 여기서 ys_lens가 사용됨!

        if torch.isnan(loss).any():
            logging.warning("❌ CTC Loss에서 NaN 발견! 0으로 대체합니다.")
            loss = torch.zeros_like(loss)
        
        self.loss = loss # 인스턴스 변수에 손실 값 저장

        logging.info(
            f"{self.__class__.__name__} "
            f"output lengths: {ys_lens.tolist()} " # ys_lens를 로깅
            f"input lengths: {hlens.tolist()}"
        )
        
        # 6. 최종 손실 리덕션
        if self.reduce and self.loss_fn.reduction == 'none': # loss_fn의 reduction이 'none'일 때만 sum
            self.loss = self.loss.sum()
            logging.info(f"ctc loss (summed): {float(self.loss)}")
        elif self.loss_fn.reduction != 'none': # reduction이 이미 적용된 경우
            logging.info(f"ctc loss (reduced by CTCLoss): {float(self.loss)}")

        return self.loss
    
    
    def softmax(self, hs_pad):
        """softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        self.probs = F.softmax(self.ctc_lo(hs_pad), dim=2)
        return self.probs

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: argmax applied 2d tensor (B, Tmax)
        :rtype: torch.Tensor
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)
    
    def forced_align(self, h, y, blank_id=0):
        """forced alignment.
        
        :param torch.Tensor h: hidden state sequece, 2d tensor (T, D)
        :param torch.Tensor y: id sequence tensor 1d tensor (L)
        :param int y: blank symbol index
        :return: best alignment results
        :rtype: list
        """

        def interpolate_blank(label, blank_id=0):
            """Insert blank token between every two label token."""
            label = np.expand_dims(label, 1)
            blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
            label = np.concatenate([blanks, label], axis=1)
            label = label.reshape(-1)
            label = np.append(label, label[0])  # two label token
            return label

        lpz = self.log_softmax(h)
        lpz = lpz.squeeze(0)
        
        y_int = interpolate_blank(y, blank_id)
        
        logdelta = np.zeros((lpz.size(0), len(y_int))) - 100000000000.0
        state_path = (
            np.zeros((lpz.size(0), len(y_int)), dtype=np.int16) - 1
        )
        
        logdelta[0, 0] = lpz[0][y_int[0]]
        logdelta[0, 1] = lpz[0][y_int[1]]
        
        for t in range(1, lpz.size(0)):
            for s in range(len(y_int)):
                if y_int[s] == blank_id or s < 2 or y_int[s] == y_int[s - 2]:
                    candidates = np.array([logdelta[t - 1, s], logdelta[t - 1, s - 1]])
                    prev_state = [s, s - 1]
                else:
                    candidates = np.array(
                        [
                            logdelta[t - 1, s], 
                            logdelta[t - 1, s - 1], 
                            logdelta[t - 1, s - 2],
                        ]
                    )
                    prev_state = [s, s - 1, s - 2]
                logdelta[t, s] = np.max(candidates) + lpz[t][y_int[s]]
                state_path[t, s] = prev_state[np.argmax(candidates)]
                
        state_seq = -1 * np.ones((lpz.size(0), 1), dtype=np.int16)
        
        candidates = np.array(
            [logdelta[-1, len(y_int) - 1], logdelta[-1, len(y_int) - 2]]
        )
        prev_state = [len(y_int) - 1, len(y_int - 2)]
        state_seq[-1] = prev_state[np.argmax(candidates)]
        for t in range(lpz.size(0) - 2, -1, -1):
            state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]
        
        output_state_seq = []
        for t in range(0, lpz.size(0)):
            output_state_seq.append(y_int[state_seq[t, 0]])
            
        return output_state_seq
    


    def ctc_for(args, odim, reduce=True):
        """Returns the CTC module for the given args and output dimension

        :param Namespace args: the program args
        :param int odim : The output dimension
        :param bool reduce : return the CTC loss in a scalar
        :return: the corresponding CTC module
        """
        num_encs = getattr(args, "num_encs", 1)  # use getattr to keep compatibility
        if num_encs == 1:
            # compatible with single encoder asr mode
            return CTC(
                odim, args.eprojs, args.dropout_rate, ctc_type=args.ctc_type, reduce=reduce
            )
        elif num_encs >= 1:
            ctcs_list = torch.nn.ModuleList()
            if args.share_ctc:
                # use dropout_rate of the first encoder
                ctc = CTC(
                    odim,
                    args.eprojs,
                    args.dropout_rate[0],
                    ctc_type=args.ctc_type,
                    reduce=reduce,
                )
                ctcs_list.append(ctc)
            else:
                for idx in range(num_encs):
                    ctc = CTC(
                        odim,
                        args.eprojs,
                        args.dropout_rate[idx],
                        ctc_type=args.ctc_type,
                        reduce=reduce,
                    )
                    ctcs_list.append(ctc)
            return ctcs_list
        else:
            raise ValueError(
                "Number of encoders needs to be more than one. {}".format(num_encs)
            )