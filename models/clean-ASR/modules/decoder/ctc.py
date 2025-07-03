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
        
        self.ctc_type = self.ctc_type if V(torch.__version__) < V("1.7.0") else "builtin"
        
        if self.ctc_type != self.ctc_type:
            logging.warning(f"CTC was set to {self.ctc_type} due to PyTorch version.")
            
        if self.ctc_type == "builtin":
            reduction_type = "sum" if self.reduce else "none"
            self.ctc_loss = torch.nn.CTCLoss(
                reduction=reduction_type, zero_infinity=True
            )
        elif self.ctc_type == "cudnnctc":
            reduction_type = "sum" if self.reduce else "none"
            self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)
        elif self.ctc_type == "gtnctc":
            from espnet.nets.pytorch_backend.gtn_ctc import GTNCTCLossFunction

            self.ctc_loss = GTNCTCLossFunction.apply
        else:
            raise ValueError(
                'ctc_type must be "builtin" or "gtnctc": {}'.format(self.ctc_type)
            )
        self.ignore_id = 0
        self.reduce = self.reduce
        
    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        if self.ctc_type == "builtin":
            # if th_ilen < 2 * th_olen - 1:
            #     print(f"❌ invalid CTC condition: input {th_ilen} vs target {th_olen}")

            th_pred = th_pred.log_softmax(2)
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            size = th_pred.size(1)

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        elif self.ctc_type == "gtnctc":
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, th_target, th_ilen, 0, "none")

        else:
            raise NotImplementedError
        
    def forward(self, hs_pad, hlens, ys_pad):
        """CTC forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequeces (B)
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        """
        # Check CTC impossibility
        # print(f"[DEBUG] CTC input - hs_pad: {hs_pad.shape}, hlens: {hlens.shape}, ys_pad: {ys_pad.shape}")
        ys_lens = torch.tensor([len(y[y != self.ignore_id]) for y in ys_pad], device=hlens.device)
        invalid_mask = hlens < (2 * ys_lens - 1)
        # if invalid_mask.any():
        #     print("❗ CTC 조건을 만족하지 않는 샘플 있음!")
        #     print("hlens :", hlens[invalid_mask])
        #     print("ys_lens:", ys_lens[invalid_mask])

        ys = [y[y != self.ignore_id] for y in ys_pad]  # 각 샘플의 유효한 label 길이
        ys_lens = torch.tensor([len(y) for y in ys], device=hlens.device)

        # ✅ CTC 조건 점검
        invalid_mask = hlens < (2 * ys_lens - 1)
        # if invalid_mask.any():
        #     print("❗ CTC 조건 위반 샘플 있음")
        #     print("    - hlens:   ", hlens[invalid_mask].tolist())
        #     print("    - ys_lens:", ys_lens[invalid_mask].tolist())

        ys = [y[y != self.ignore_id] for y in ys_pad]
        
        # zero padding for hs
        ys_hat = self.ctc_lo(self.dropout(hs_pad))
        if self.ctc_type != "gtnctc":
            ys_hat = ys_hat.transpose(0, 1)
        
        # print(f"[DEBUG] CTC logits shape: {ys_hat.shape}, min={ys_hat.min().item()}, max={ys_hat.max().item()}")
        
        if self.ctc_type != "builtin":
            ys_hat = torch.clamp(ys_hat, min=-1e8, max=1e8)
            olens = to_device(ys_hat, torch.LongTensor([len(s) for s in ys]))
            hlens = hlens.long()
            ys_pad = torch.cat(ys)
            self.loss = self.loss_fn(ys_hat, ys_pad, hlens, olens)
            if torch.isnan(loss).any():
                print("❌ CTC Loss에서 NaN 발견! 0으로 대체합니다.")
                loss = torch.zeros_like(loss)
            self.loss = loss
        else:
            self.loss = None
            hlens = torch.tensor(hlens, dtype=torch.int32, device=hs_pad.device)
            olens = torch.tensor(
                [y.size(0) for y in ys], dtype=torch.int32, device=hs_pad.device
            )
            # zero padding for ys
            ys_true = torch.cat(ys).cpu().int()  # batch x olen
            # get ctc loss
            # expected shape of seqLength x batchSize x vocab_size
            dtype = ys_hat.dtype
            if self.ctc_type == "cudnnctc":
                # use GPU when using the cuDNN implementation
                ys_true = to_device(hs_pad, ys_true)
            if self.ctc_type == "gtnctc":
                # keep as list for gtn
                ys_true = ys
            self.loss = to_device(
                hs_pad, self.loss_fn(ys_hat, ys_true, hlens, olens)
            ).to(dtype=dtype)
            
        logging.info(
            self.__class__.__name__
            + " output lengths: "
            + "".join(str(olens).split("\n"))
        )
        
        if self.reduce:
            self.loss = self.loss.sum()
            logging.info("ctc loss:" + str(float(self.loss)))
            
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