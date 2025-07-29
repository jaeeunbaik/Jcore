import sentencepiece as spm
import argparse
import os

def extract_and_convert_tokens_to_text(scp_path: str, sp_model_path: str, output_txt_path: str):
    """
    Reads token IDs from an SCP file, converts them to text using SentencePiece,
    and saves the text to an output file.

    Args:
        scp_path (str): Path to the input .scp file (e.g., train_token.scp).
        sp_model_path (str): Path to the SentencePiece model file (.model).
        output_txt_path (str): Path to the output .txt file.
    """
    # if not os.path.exists(scp_path):
    #     print(f"Error: Input SCP file not found at {scp_path}")
    #     return

    # if not os.path.exists(sp_model_path):
    #     print(f"Error: SentencePiece model file not found at {sp_model_path}")
    #     return

    # # SentencePiece 모델 로드
    # sp = spm.SentencePieceProcessor()
    # sp.load(sp_model_path)
    # print(f"SentencePiece model loaded from: {sp_model_path}")

    # 결과 텍스트를 저장할 파일 열기
    with open(output_txt_path, 'w', encoding='utf-8') as outfile:
        # SCP 파일 읽기
        with open(scp_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                line = line.strip()
                if not line: # 빈 라인 건너뛰기
                    continue

                try:
                    # '|'를 기준으로 분리하고 오른쪽 토큰 문자열 가져오기
                    parts = line.split('|')
                    if len(parts) < 2:
                        print(f"Warning: Line {line_num + 1} in {scp_path} does not contain '|' delimiter or tokens. Skipping: {line}")
                        continue
                    
                    text = parts[1].strip()
                    # if not token_str: # 토큰 부분이 비어있는 경우
                    #     print(f"Warning: Line {line_num + 1} has empty token string. Skipping: {line}")
                    #     continue

                    # # 토큰 문자열을 공백으로 분리하여 정수 리스트로 변환
                    # token_ids = list(map(int, token_str.split(' ')))
                    
                    # # SentencePiece를 사용하여 토큰 ID를 텍스트로 변환
                    # # sp.decode()는 공백 토큰을 처리하고 텍스트로 변환해줍니다.
                    # text = sp.decode(token_ids)
                    
                    # 변환된 텍스트를 출력 파일에 쓰기
                    outfile.write(text + '\n')

                except ValueError as ve:
                    print(f"Error processing line {line_num + 1} (ValueError: {ve}). Skipping: {line}")
                except IndexError as ie:
                    print(f"Error processing line {line_num + 1} (IndexError: {ie}). Skipping: {line}")
                except Exception as e:
                    print(f"An unexpected error occurred at line {line_num + 1} ({type(e).__name__}: {e}). Skipping: {line}")

    print(f"\nText extraction complete.")
    print(f"Processed tokens from {scp_path} and saved to {output_txt_path}")

if __name__ == "__main__":
    scp_file = '/home/hdd2/jenny/ASRToolkit/Jcore/scp/kspon1234_aihub비대면PAPBHA/train.scp'
    sp_model = '/home/hdd2/jenny/ASRToolkit/Self-Distillation-ASR/util/spm/bpe/kor_bpe5000.model'
    output_file = '/home/hdd2/jenny/ASRToolkit/Jcore/scp/kspon1234_aihub비대면PAPBHA/input.txt'

    extract_and_convert_tokens_to_text(scp_file, sp_model, output_file)