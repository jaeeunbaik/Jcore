import os
import sentencepiece as spm
from typing import Dict, List, Tuple
import num2words
import re


def number_to_korean(text):
    """
    문자열 내의 모든 숫자를 한국어 발음으로 변환합니다.
    """
    def replace_number(match):
        num = int(match.group(0))
        return num2words.num2words(num, to='cardinal', lang='ko')

    return re.sub(r'\d+', replace_number, text)



def update_path(input_file_path: str, output_file_path: str, new_base_path: str):
    """
    train.scp 파일의 오디오 경로를 업데이트합니다.
    오디오 경로 내에 "Training/"이 포함되어 있으면,
    해당 부분 이후의 경로(relative path)를 new_base_path와 결합하여 새로운 경로로 변경합니다.
    나머지 텍스트(스크립트)는 그대로 유지합니다.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in, \
             open(output_file_path, 'w', encoding='utf-8') as f_out:
             
            for line_num, line in enumerate(f_in, start=1):
                line = line.strip()
                if not line:
                    continue

                # 만약 라인이 '|'를 포함한다면 오디오 경로와 스크립트를 분리합니다.
                parts = line.split('|')
                original_audio_path = parts[0]

                if "비대면_진료를_위한_의료진_및_환자_음성/" in original_audio_path:
                    print(f'{original_audio_path}처리중')
                    idx = original_audio_path.find("비대면_진료를_위한_의료진_및_환자_음성/") + len("비대면_진료를_위한_의료진_및_환자_음성/")
                    relative_path = original_audio_path[idx:]
                    new_full_audio_path = os.path.join(new_base_path, relative_path)
                else:
                    print(f"경고: 라인 {line_num} - '비대면_진료를_위한_의료진_및_환자_음성/'를 찾을 수 없습니다. 원본 경로 사용: {original_audio_path}")
                    new_full_audio_path = original_audio_path

                # 스크립트가 존재한다면 그대로 유지, 없으면 빈 문자열 사용
                korean_script = parts[1] if len(parts) > 1 else ""
                f_out.write(f"{new_full_audio_path}|{korean_script}\n")
        
        print(f"오디오 경로 업데이트 완료. '{output_file_path}' 파일 작성 완료.")
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다 - {e}")
    except Exception as e:
        print(f"오류 발생: {e}")




def txt_to_token(input_file_path: str, output_file_path: str, spm_model_path: str):
    """
    train.scp 파일의 각 줄에서 한글 스크립트를 SentencePiece를 이용해 토큰 ID 시퀀스로 변환합니다.
    오디오 경로는 그대로 유지하며, 변환된 토큰 ID 시퀀스로 새 파일에 작성합니다.

    Args:
        input_file_path (str): 원본 train.scp 파일의 전체 경로.
        output_file_path (str): 토큰 ID 시퀀스로 변환된 결과를 저장할 파일의 전체 경로.
        spm_model_path (str): SentencePiece 모델 파일 (.model)의 전체 경로.
    """
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(spm_model_path)
        print(f"SentencePiece 모델 '{spm_model_path}' 로드 완료.")

        with open(input_file_path, 'r', encoding='utf-8') as f_in, \
             open(output_file_path, 'w', encoding='utf-8') as f_out:
             
            for line_num, line in enumerate(f_in, start=1):
                if '|' in line: # 음원 - 텍스트 pair scp 파일일 때
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('|')
                    if len(parts) < 2:
                        print(f"경고: 라인 {line_num} - 유효하지 않은 형식: {line}.")
                        continue
                        
                    original_audio_path = parts[0]
                    korean_script = '|'.join(parts[1:])
                    
                    token_ids = sp.encode_as_ids(korean_script)
                    token_ids_str = ' '.join(map(str, token_ids))
                    
                    f_out.write(f"{original_audio_path}|{token_ids_str}\n")
                else: # 텍스트만 존재하는 scp 파일일 때
                    token_ids = sp.encode_as_ids(line)
                    token_ids_str = ' '.join(map(str, token_ids))
                    
                    f_out.write(f"{token_ids_str}\n")
                    
        print(f"문자열을 토큰 ID 시퀀스로 변환 완료. '{output_file_path}' 파일 작성 완료.")
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다 - {e}")
    except Exception as e:
        print(f"오류 발생: {e}")
        
        
        
def convert_num2word(input_file_path: str, output_file_path: str):
    """
    train.scp 파일의 각 줄에서 한글 스크립트를 숫자를 한국어 발음으로 변환하여 저장합니다.
    오디오 경로는 그대로 유지합니다.

    Args:
        input_file_path (str): 원본 train.scp 파일의 전체 경로.
        output_file_path (str): 숫자가 변환된 결과를 저장할 파일의 전체 경로.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in, \
             open(output_file_path, 'w', encoding='utf-8') as f_out:
             
            for line_num, line in enumerate(f_in, start=1):
                if '|' in line: # 음원 - 텍스트 pair scp 파일일 때
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('|')
                    if len(parts) < 2:
                        print(f"경고: 라인 {line_num} - 유효하지 않은 형식: {line}.")
                        continue
                        
                    original_audio_path = parts[0]
                    korean_script = '|'.join(parts[1:])
                    
                    # 숫자 -> 한글 발음 변환 적용
                    korean_script = number_to_korean(korean_script)
                    
                    f_out.write(f"{original_audio_path}|{korean_script}\n")
                else: # 텍스트만 존재하는 scp 파일일 때
                    line = line.strip()
                    # 숫자 -> 한글 발음 변환 적용
                    line = number_to_korean(line)
                    
                    f_out.write(f"{line}\n")
                    
        print(f"문자열을 숫자를 한글로 변환 완료. '{output_file_path}' 파일 작성 완료.")
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다 - {e}")
    except Exception as e:
        print(f"오류 발생: {e}")

        
        
def truncate(input_file_path: str, output_file_path: str):
    """
    train.scp 파일에서 각 줄의 오디오 파일 경로 상의 폴더명이 PA, HA, HB로 시작하는 라인만 필터링하여,
    해당 라인을 새 파일에 그대로 저장합니다.

    Args:
        input_file_path (str): 원본 train.scp 파일의 전체 경로.
        output_file_path (str): 필터링된 결과를 저장할 파일의 전체 경로.
    """
    allowed_prefixes = ['PA', 'PB', 'PC', 'HA', 'HB']
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in, \
             open(output_file_path, 'w', encoding='utf-8') as f_out:
             
            for line_num, line in enumerate(f_in, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('|')
                if len(parts) < 1:
                    print(f"경고: 라인 {line_num} - 유효하지 않은 형식입니다: '{line}'. 이 줄은 건너뜁니다.")
                    continue

                original_audio_path = parts[0]
                folder = os.path.basename(os.path.dirname(original_audio_path))
                
                prefix = folder[:2]
                # prefix = os.path.dirname(original_audio_path).split('/')[-2]
                if prefix in allowed_prefixes:
                    f_out.write(line + "\n")
                else:
                    print(f"라인 {line_num}: '{prefix}' 폴더는 허용되지 않습니다. 해당 라인을 건너뜁니다.")

        print(f"필터링 완료. '{output_file_path}' 파일 작성 완료.")

    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다 - {e}")
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")


# --- 사용 예시 ---
if __name__ == "__main__":
    # 1. 원본 train.scp 파일의 경로를 여기에 입력하세요.
    original_scp_file = '/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/scp/AIHub_비대면진료/train.scp'
    
    # 2. 새로 생성될 train.scp 파일의 경로를 여기에 입력하세요.
    update_scp_file = '/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/scp/AIHub_비대면진료/train_new.scp' 
    output_scp_file = '/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/scp/kspon_aihub비대면/dev.scp'
    out_scp_file = '/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/scp/kspon_aihub비대면/dev_token.scp'
    # 3. 새로운 데이터셋 기본 경로를 여기에 입력하세요.
    new_base_directory = '/home/hdd1/jenny/AIHub_비대면진료'

    # 4. 학습된 SentencePiece 모델 파일 (.model)의 경로를 여기에 입력하세요.
    spm_model_file = '/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/util/spm/bpe/kor_bpe5000.model' 


    print(f"'{original_scp_file}' 파일을 처리하여 토큰화된 '{output_scp_file}'을 생성합니다...")
    # update_path(original_scp_file, update_scp_file, new_base_directory)
    # convert_num2word(original_scp_file, update_scp_file)
    # truncate(update_scp_file, output_scp_file)
    txt_to_token(output_scp_file, out_scp_file, spm_model_file)
    # --- 업데이트된 파일 내용 확인 (선택 사항) ---
    print(f"\n--- 업데이트된 '{output_scp_file}' 파일 내용 ---")
    try:
        with open(output_scp_file, 'r', encoding='utf-8') as f:
            for line in f:
                print(line.strip())
    except FileNotFoundError:
        print("생성된 파일을 열 수 없습니다.")
