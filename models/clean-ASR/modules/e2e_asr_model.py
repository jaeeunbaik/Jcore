"""
🖤🐰 Jaeeun Baik, 2025
"""

import numpy
import logging
import torch
import torch.nn as nn
import sentencepiece as spm

from argparse import Namespace

from modules.encoder.conformer_encoder import ConformerEncoder
from modules.decoder.transformer_decoder import TransformerDecoder
from modules.decoder.ctc import CTC
from modules.decoder.rnn_transducer import RNNTransducer
from modules.decoder.label_smoothing_loss import LabelSmoothingLoss
from modules.transformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention

from util.utils_text import ErrorCalculator, add_sos_eos
from util.utils_module import subsequent_mask, target_mask, end_detect, th_accuracy
from util.utils_ctc import CTCPrefixScore, CTCPrefixScorer

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5

class e2eASR(nn.Module):
    def __init__(
        self,
        model_config,
        ignore_id = 0
    ):
        super().__init__()
        self.encoder_config = model_config.encoder
        self.decoder_config = model_config.decoder
        self.odim = self.decoder_config.odim 
        self.adim = self.encoder_config.output_size
        self.ignore_id = ignore_id
        
        # Encoder
        if self.encoder_config.type == "Conformer":
            self.encoder = ConformerEncoder(self.encoder_config)
        
        # Decoder
        self.decoder_type = self.decoder_config.type
        
        if self.decoder_type == "rnnt":
            self.rnnt = RNNTransducer(att=RelPositionMultiHeadedAttention(
                                             n_head=self.encoder_config.attention_heads,
                                             n_feat=self.odim, 
                                             dropout_rate=0.1),
                                         eprojs=self.decoder_config.eprojs,
                                         odim=self.decoder_config.odim,
                                         dtype=self.decoder_config.dtype,
                                         dlayers=self.decoder_config.dlayers,
                                         dunits=self.decoder_config.dunits,
                                        )
        elif self.decoder_type in ["transformer", "hybrid"]:
            self.decoder = TransformerDecoder(self.decoder_config.odim)
            self.criterion = LabelSmoothingLoss(self.decoder_config.odim, self.ignore_id, 0.1, False)
        if self.decoder_type in ["ctc", "hybrid"]:
            self.ctc = CTC(self.decoder_config)
        self.ctc_weight = self.decoder_config.ctc_weight
        
        self.blank = 0
        self.sos = 2
        self.eos = 3
        self.ignore_id = ignore_id
        
        if model_config.report_cer or model_config.report_wer:
            self.error_calculator = ErrorCalculator(
                model_config.char_list,
                model_config.sym_space,
                model_config.sym_blank,
                model_config.report_cer,
                model_config.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None


    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        hs_pad, hs_mask = self.encoder(xs_pad, ilens)
        
        # print(f"[DEBUG] Encoder output: shape={hs_pad.shape}, dtype={hs_pad.type}, device={hs_pad.device}")
        # print(f"[DEBUG] Target shape={ys_pad.shape}, dtype={ys_pad.dtype}, device={ys_pad.device}")
        self.hs_pad = hs_pad
        
        # 2. decoder loss
        if self.decoder_type in ["transformer", "hybrid"]:
            ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad, self.sos, self.eos, self.ignore_id
            )
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
            self.pred_pad = pred_pad

            # 3. compute attention loss
            loss_att = self.criterion(pred_pad, ys_out_pad)
            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        else:
            loss_att = None
            self.acc = None

        # 3. ctc loss
        cer_ctc = None
        if self.decoder_type in ["transformer", "rnnt"]:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(self.hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if not self.training and self.error_calculator is not None:
                ys_hat = self.ctc.argmax(self.hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(self.hs_pad)
        
        # 4. rnnt loss
        if self.decoder_type == 'rnnt':
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_rnnt, acc, ppl = self.rnnt(hs_pad, hs_len, ys_pad)

        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # 5. return loss
        if self.decoder_type == "hybrid":
            loss = (1 - self.ctc_weight) * loss_att + self.ctc_weight * loss_ctc
        elif self.decoder_type == "ctc":
            loss = loss_ctc
        elif self.decoder_type == "transformer":
            loss = loss_att
        elif self.decoder_type == "rnnt":
            loss = loss_rnnt

        return {
            "loss": loss,
            "loss_ctc": loss_ctc.item() if loss_ctc is not None else None, 
            "loss_att": loss_att.item() if loss_att is not None else None, 
            "cer": cer,
            "wer": wer,
            "acc": acc if acc is not None else None,
            "ppl": ppl if ppl is not None else None
        }
    
    

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x, ilens):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x)
        # x = torch.as_tensor(x).unsqueeze(0)
        enc_output, *_ = self.encoder(x, ilens)
        return enc_output.squeeze(0)

    def recognize(self, x, ilens, y, recog_args, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        # print(f"[DEBUG] input shape {x.shape}")
        enc_output = self.encode(x, ilens)
        print(f"[DEBUG] encoder output {enc_output.shape}")
        if enc_output.dim() == 2:
            enc_output = enc_output.unsqueeze(0)
        
        if self.decoder_type == 'ctc':
            lpz = self.ctc.log_softmax(enc_output)  # 시퀀스의 각 시간 단계에서 가장 높은 확률을 가진 토큰 ID
            print(f"[DEBUG] ctc decoding output {lpz.shape}")
            lpz = lpz.squeeze(0)
            
            best_paths = torch.argmax(lpz, dim=2)
            print(f"[DEBUG] best path of ctc decoding {best_paths.shape}")
            best_paths_list = best_paths.cpu().tolist()

            nbest_hyps = []
            for batch_idx, best_path in enumerate(best_paths_list):
                hyp = [self.sos]
                prev_token = None
                for token in best_path:
                    if token != prev_token and token != self.blank:
                        hyp.append(token)
                    prev_token = token
                nbest_hyps.append({"score": 0.0, "yseq": hyp})
            return nbest_hyps
            
        elif self.decoder_type == 'hybrid':  # Hybrid
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
            
        elif self.decoder_type == 'rnnt':
            sp = spm.SentencePieceProcessor()
            sp.load("/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/util/spm/unigram/unigram5000.model")
            char_list = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
            lpz = self.ctc.log_softmax(enc_output)
            recog_args = Namespace(
                beam_size=5, 
                penalty=0.3,
                ctc_weight=0.0,
                maxlenratio=1.0,
                minlenratio=0.1,
                nbest=5
            )
            nbest_hyps = self.rnnt.recognize_beam_batch(enc_output, lpz, ilens, recog_args, char_list)
            return nbest_hyps
        
        
        elif self.decoder_type == 'transformer':
            lpz = None

            h = enc_output.squeeze(0)

            logging.info("input lengths: " + str(h.size(0)))
            # search parms
            beam = recog_args.beam_size
            penalty = recog_args.penalty
            ctc_weight = recog_args.ctc_weight

            # preprare sos
            y = self.sos
            vy = h.new_zeros(1).long()

            if recog_args.maxlenratio == 0:
                maxlen = h.shape[0]
            else:
                # maxlen >= 1
                maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
            minlen = int(recog_args.minlenratio * h.size(0))
            logging.info("max output length: " + str(maxlen))
            logging.info("min output length: " + str(minlen))

            # initialize hypothesis
            if rnnlm:
                hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
            else:
                hyp = {"score": 0.0, "yseq": [y]}
            if lpz is not None:
                ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
                hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
                hyp["ctc_score_prev"] = 0.0
                if ctc_weight != 1.0:
                    # pre-pruning based on attention scores
                    ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
                else:
                    ctc_beam = lpz.shape[-1]
            hyps = [hyp]
            ended_hyps = []

            traced_decoder = None
            for i in range(maxlen):
                logging.debug("position " + str(i))

                hyps_best_kept = []
                for hyp in hyps:
                    vy[0] = hyp["yseq"][i]

                    # get nbest local scores and their ids
                    ys_mask = subsequent_mask(i + 1).unsqueeze(0).to(vy.device)
                    ys = torch.tensor(hyp["yseq"]).unsqueeze(0).to(vy.device)
                    # FIXME: jit does not match non-jit result
                    if use_jit:
                        if traced_decoder is None:
                            traced_decoder = torch.jit.trace(
                                self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                            )
                        local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                    else:
                        local_att_scores = self.decoder.forward_one_step(
                            ys, ys_mask, enc_output
                        )[0]

                    if rnnlm:
                        rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                        local_scores = (
                            local_att_scores + recog_args.lm_weight * local_lm_scores
                        )
                    else:
                        local_scores = local_att_scores

                    if lpz is not None:
                        local_best_scores, local_best_ids = torch.topk(
                            local_att_scores, ctc_beam, dim=1
                        )
                        ctc_scores, ctc_states = ctc_prefix_score(
                            hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                        )
                        local_scores = (1.0 - ctc_weight) * local_att_scores[
                            :, local_best_ids[0]
                        ] + ctc_weight * torch.from_numpy(
                            ctc_scores - hyp["ctc_score_prev"]
                        )
                        if rnnlm:
                            local_scores += (
                                recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                            )
                        local_best_scores, joint_best_ids = torch.topk(
                            local_scores, beam, dim=1
                        )
                        local_best_ids = local_best_ids[:, joint_best_ids[0]]
                    else:
                        local_best_scores, local_best_ids = torch.topk(
                            local_scores, beam, dim=1
                        )

                    for j in range(beam):
                        new_hyp = {}
                        new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                        new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                        new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                        new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                        if rnnlm:
                            new_hyp["rnnlm_prev"] = rnnlm_state
                        if lpz is not None:
                            new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                            new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                        # will be (2 x beam) hyps at most
                        hyps_best_kept.append(new_hyp)

                    hyps_best_kept = sorted(
                        hyps_best_kept, key=lambda x: x["score"], reverse=True
                    )[:beam]

                # sort and get nbest
                hyps = hyps_best_kept
                logging.debug("number of pruned hypothes: " + str(len(hyps)))

                # add eos in the final loop to avoid that there are no ended hyps
                if i == maxlen - 1:
                    logging.info("adding <eos> in the last position in the loop")
                    for hyp in hyps:
                        hyp["yseq"].append(self.eos)

                # add ended hypothes to a final list, and removed them from current hypothes
                # (this will be a probmlem, number of hyps < beam)
                remained_hyps = []
                for hyp in hyps:
                    if hyp["yseq"][-1] == self.eos:
                        # only store the sequence that has more than minlen outputs
                        # also add penalty
                        if len(hyp["yseq"]) > minlen:
                            hyp["score"] += (i + 1) * penalty
                            if rnnlm:  # Word LM needs to add final <eos> score
                                hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                    hyp["rnnlm_prev"]
                                )
                            ended_hyps.append(hyp)
                    else:
                        remained_hyps.append(hyp)

                # end detection
                if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                    logging.info("end detected at %d", i)
                    break

                hyps = remained_hyps
                if len(hyps) > 0:
                    logging.debug("remeined hypothes: " + str(len(hyps)))
                else:
                    logging.info("no hypothesis. Finish decoding.")
                    break

                logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

            nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
                : min(len(ended_hyps), recog_args.nbest)
            ]

            # check number of hypotheis
            if len(nbest_hyps) == 0:
                logging.warning(
                    "there is no N-best results, perform recognition "
                    "again with smaller minlenratio."
                )
                # should copy becasuse Namespace will be overwritten globally
                recog_args = Namespace(**vars(recog_args))
                recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
                return self.recognize(x, recog_args, rnnlm)

            logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
            logging.info(
                "normalized log probability: "
                + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
            )
        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
            # if isinstance(m, DynamicConvolution2D):
            #     ret[name + "_time"] = m.attn_t.cpu().numpy()
            #     ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.ctc_weight == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret

    def infer(self, x):
        self.eval()
        with torch.no_grad():
            enc_out, *_ = self.encoder(x.unsqueeze(0), None)
            enc_out = enc_out.squeeze(0)
            result = {}
            if hasattr(self, "ctc"):
                result["ctc_out"] = self.ctc.decode(enc_out)
            if hasattr(self, "decoder"):
                result["dec_out"] = self.decoder.decode(enc_out)
            return result

