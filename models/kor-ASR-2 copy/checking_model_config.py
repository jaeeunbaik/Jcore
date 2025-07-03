import torch
import time
import yaml
# from ptflops import get_model_complexity_info # ptflops 대신 thop 사용
from torch import nn
from thop import profile, clever_format # thop 라이브러리 임포트
from modules.e2e_asr_model import e2eASR
from attrdict import AttrDict

# Wrapper 모델 정의 (ptflops/thop 호환을 위해)
class FlopsCompatibleASR(e2eASR):
    def __init__(self, model_config, tokenizer_path, decoder_type, ignore_id=0):
        super().__init__(model_config, tokenizer_path, decoder_type, ignore_id)

    # 이 forward 메서드는 thop이나 ptflops가 모델의 FLOPs를 계산할 때만 사용됩니다.
    # 인자 개수를 맞추기 위해 xs_pad만 받고, 나머지는 내부에서 더미로 생성하여 super().forward()에 전달합니다.
    def forward(self, xs_pad): # xs_pad만 받음
        batch_size = xs_pad.size(0)
        
        # MACs/FLOPs 계산 시점에 필요한 더미 ys_pad, ylens, ilens
        max_target_length = xs_pad.size(1) // 4
        if max_target_length == 0:
            max_target_length = 1

        odim = self.odim 

        dummy_ys_pad = torch.randint(
            low=4, high=odim, size=(batch_size, max_target_length), dtype=torch.int32, device=xs_pad.device
        )
        dummy_ylens = torch.randint(
            low=1, high=max_target_length + 1, size=(batch_size,), dtype=torch.int32, device=xs_pad.device
        )
        dummy_ilens = torch.tensor([xs_pad.size(1)] * batch_size, dtype=torch.int32, device=xs_pad.device)
        
        # 원래 e2eASR의 forward 메서드를 호출
        return super().forward(xs_pad, dummy_ilens, dummy_ys_pad, dummy_ylens)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return AttrDict(config)

def measure_inference_time(model_original_e2e_asr, dummy_input, device="cuda"):
    """
    Measure inference time for a given model and input.
    이 함수는 원래 e2eASR 모델의 forward 시그니처에 맞춰 작동합니다.
    """
    model_original_e2e_asr.eval()
    model_original_e2e_asr.to(device)
    dummy_input = dummy_input.to(device)

    batch_size = dummy_input.size(0)
    max_target_length = dummy_input.size(1) // 4
    if max_target_length == 0:
        max_target_length = 1

    odim = model_original_e2e_asr.odim

    ys_pad = torch.randint(
        low=4, high=odim, size=(batch_size, max_target_length), dtype=torch.int32, device=dummy_input.device
    )
    ylens = torch.randint(
        low=1, high=max_target_length + 1, size=(batch_size,), dtype=torch.int32, device=dummy_input.device
    )
    ilens = torch.tensor([dummy_input.size(1)] * batch_size, dtype=torch.int32, device=dummy_input.device)

    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            # 이 model_original_e2e_asr는 e2eASR 인스턴스입니다.
            model_original_e2e_asr(dummy_input, ilens, ys_pad, ylens)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        model_original_e2e_asr(dummy_input, ilens, ys_pad, ylens)
    end_time = time.time()

    return end_time - start_time

def compare_models(config_path):
    """Compare models with different decoders."""
    config = load_config(config_path)
    encoder_dim = config.model.asr.encoder.encoder_dim
    input_dim = config.model.asr.encoder.input_dim
    odim = config.model.asr.decoder.odim

    # 타이밍 측정 및 FLOPs 계산의 xs_pad로 사용할 더미 입력
    dummy_input_for_xs_pad = torch.randn(1, 10 * config.data.sample_rate // config.data.hop_length, input_dim)

    results = []

    for decoder_type in ["ctc", "transformer", "rnnt"]:
        print(f"Evaluating model with {decoder_type} decoder...")
        config.model.asr.decoder.type = decoder_type
        
        # FLOPs 계산을 위한 래퍼 모델 (FlopsCompatibleASR)
        model_flops = FlopsCompatibleASR(config.model.asr, config.data.tokenizer, decoder_type=decoder_type)
        # 실제 추론 시간 측정을 위한 원래 모델 (e2eASR)
        model_inference = e2eASR(config.model.asr, config.data.tokenizer, decoder_type=decoder_type)


        # Measure parameter count (둘 중 아무 모델이나 사용 가능, 파라미터는 동일)
        param_count = sum(p.numel() for p in model_flops.parameters())

        # Measure MACs using thop
        # thop.profile은 input_res를 튜플 형태로 받습니다.
        # FlopsCompatibleASR의 forward(xs_pad)에 맞춰 xs_pad의 shape만 전달
        # FlopsCompatibleASR의 forward 메서드는 xs_pad만 받으므로,
        # dummy_input_for_xs_pad를 직접 전달합니다.
        
        # thop의 profile 함수는 모델과 입력 튜플을 받습니다.
        # 이 때 입력 튜플은 모델의 forward 메서드의 인자 순서와 형태를 따라야 합니다.
        # FlopsCompatibleASR의 forward(xs_pad)에 맞춰 (xs_pad,) 형태로 전달
        
        # thop에 전달할 입력 텐서 (xs_pad만)
        # 이 텐서는 model_flops의 forward 메서드에 전달될 실제 텐서입니다.
        inputs_for_thop = (dummy_input_for_xs_pad,) 
        macs, params = profile(model_flops, inputs=inputs_for_thop, verbose=False)


        # Measure inference time (원래 e2eASR 모델 사용)
        # measure_inference_time 함수로 원래의 e2eASR 인스턴스를 전달합니다.
        inference_time = measure_inference_time(model_inference, dummy_input_for_xs_pad)

        results.append({
            "decoder_type": decoder_type,
            "param_count": param_count,
            "macs": macs,
            "inference_time": inference_time
        })

    return results

def compare_models(config_path):
    """Compare models with different decoders."""
    config = load_config(config_path)
    encoder_dim = config.model.asr.encoder.encoder_dim
    input_dim = config.model.asr.encoder.input_dim
    odim = config.model.asr.decoder.odim

    dummy_input_for_xs_pad = torch.randn(1, 5 * config.data.sample_rate // config.data.hop_length, input_dim)

    results = []

    # thop에 전달할 custom_ops 딕셔너리 정의
    # nn.ReLU에 대한 FLOPs 계산을 0으로 정의
    # 만약 다른 활성화 함수 (예: nn.GELU, nn.SiLU 등)를 사용한다면 해당 클래스도 추가해야 합니다.
    custom_ops_dict = {
        nn.ReLU: None, # ReLU 연산은 FLOPs 0으로 처리 (thop이 기본적으로 무시하도록)
        # 만약 다른 활성화 함수가 PositionwiseFeedForward에 사용되었다면 여기에 추가
        # nn.GELU: None,
        # nn.SiLU: None,
    }

    for decoder_type in ["ctc", "transformer", "rnnt"]:
        print(f"Evaluating model with {decoder_type} decoder...")
        config.model.asr.decoder.type = decoder_type
        
        model_flops = FlopsCompatibleASR(config.model.asr, config.data.tokenizer, decoder_type)
        model_inference = e2eASR(config.model.asr, config.data.tokenizer, decoder_type)

        param_count = sum(p.numel() for p in model_flops.parameters())

        # Measure MACs using thop
        inputs_for_thop = (dummy_input_for_xs_pad,) 
        macs, params = profile(
            model_flops, 
            inputs=inputs_for_thop, 
            verbose=False,
            custom_ops=custom_ops_dict # 여기에 custom_ops 딕셔너리를 전달
        )

        inference_time = measure_inference_time(model_inference, dummy_input_for_xs_pad)

        results.append({
            "decoder_type": decoder_type,
            "param_count": param_count,
            "macs": macs,
            "inference_time": inference_time
        })

    return results

if __name__ == "__main__":
    config_path = "/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/models/clean-ASR/config_debug.yaml"
    results = compare_models(config_path)

    print("\n===== Model Comparison Results =====")
    for result in results:
        print(f"Decoder Type: {result['decoder_type']}")
        print(f"Parameter Count: {result['param_count']:,}")
        print(f"FLOPs: {result['macs'] / 1e9:.2f} GFLOPs")
        print(f"Inference Time: {result['inference_time']:.4f} seconds")
        print()