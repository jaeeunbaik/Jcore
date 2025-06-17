import os
import sentencepiece as spm
from typing import Dict, List, Tuple



def update_path(input_file_path: str, output_file_path: str, new_base_path: str):
    """
    train.scp 파일의 오디오 경로를 업데이트합니다.
    오디오 경로 내에 "비대면_진료를_위한_의료진_및_환자_음성/"이 포함되어 있으면,
    해당 부분 이후의 경로(relative path)를 new_base_path와 결합하여 새로운 경로로 변경합니다.
    나머지 텍스트(스크립트)는 그대로 유지합니다.

    Args:
        input_file_path (str): 원본 train.scp 파일의 전체 경로.
        output_file_path (str): 업데이트된 결과를 저장할 파일의 전체 경로.
        new_base_path (str): 새로운 데이터셋의 기본 경로.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in, \
             open(output_file_path, 'w', encoding='utf-8') as f_out:
             
            for line_num, line in enumerate(f_in, start=1):
                original_audio_path = line

                # 오디오 경로 업데이트
                if "Uihyeop/" in original_audio_path:
                    idx = original_audio_path.find("Uihyeop/") \
                          + len("Uihyeop/")
                    relative_path = original_audio_path[idx:]
                    new_full_audio_path = os.path.join(new_base_path, relative_path)
                else:
                    print(f"경고: 라인 {line_num} - '/home/nas/user/Uihyeop'를 찾을 수 없습니다. 원본 경로 사용: {original_audio_path}")
                    new_full_audio_path = original_audio_path

                f_out.write(f"{new_full_audio_path}")
        
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
        
        print(f"문자열을 토큰 ID 시퀀스로 변환 완료. '{output_file_path}' 파일 작성 완료.")
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
    allowed_prefixes = ['PA', 'HA', 'HB']
    
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
                if prefix in allowed_prefixes:
                    f_out.write(line + "\n")
                else:
                    print(f"라인 {line_num}: '{folder}' 폴더는 허용되지 않습니다. 해당 라인을 건너뜁니다.")

        print(f"필터링 완료. '{output_file_path}' 파일 작성 완료.")

    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다 - {e}")
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")


# --- 사용 예시 ---
if __name__ == "__main__":
    # 1. 원본 train.scp 파일의 경로를 여기에 입력하세요.
    original_scp_file = '/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/scp/rir.scp'
    
    # 2. 새로 생성될 train.scp 파일의 경로를 여기에 입력하세요.
    # 이 경로는 원본 파일과 달라야 합니다. (혹은 원본을 덮어쓰고 싶다면 동일하게 설정)
    # 안전하게 처리하기 위해서는 항상 새로운 파일로 저장하는 것을 권장합니다.
    output_scp_file = '/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/scp/rir_new.scp' 
    
    # 3. 새로운 데이터셋 기본 경로를 여기에 입력하세요.
    new_base_directory = '/home/hdd1/jenny/Uihyeop'

    # 4. 학습된 SentencePiece 모델 파일 (.model)의 경로를 여기에 입력하세요.
    spm_model_file = '/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/util/spm/bpe/kor_bpe5000.model' 


    print(f"'{original_scp_file}' 파일을 처리하여 토큰화된 '{output_scp_file}'을 생성합니다...")
    # process_scp_file_safely(original_scp_file, output_scp_file, new_base_directory, spm_model_file)
    # truncate(original_scp_file, output_scp_file)
    update_path(original_scp_file, output_scp_file, new_base_directory)
    # --- 업데이트된 파일 내용 확인 (선택 사항) ---
    print(f"\n--- 업데이트된 '{output_scp_file}' 파일 내용 ---")
    try:
        with open(output_scp_file, 'r', encoding='utf-8') as f:
            for line in f:
                print(line.strip())
    except FileNotFoundError:
        print("생성된 파일을 열 수 없습니다.")
