e2eASR
├── encoder: ConformerEncoder
├── decoder: TransformerDecoder | RNNTransducer
├── ctc: CTC
├── forward(xs, ilens, ys)
│   ├── encoder → hs_pad
│   ├── decoder → pred_pad
│   ├── loss_att, loss_ctc, acc, cer/wer
├── recognize(x)  ← beam search 포함
├── infer(x) ← 단건 디코딩
├── calculate_all_attentions
├── calculate_all_ctc_probs
├── forced_align
