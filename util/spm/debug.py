#!/usr/bin/env python3

import sys
import sentencepiece as spm
import argparse

def validate_tokenizer(model_path, text_samples):
    """토크나이저 검증"""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    print(f"토크나이저 정보:")
    print(f"- 어휘 크기: {sp.get_piece_size()}")
    
    try:
        blank_id = sp.piece_to_id('<blank>')
        print(f"- <blank> 토큰이 존재함, ID: {blank_id} (0이 되어야 함)")
    except:
        print("- <blank> 토큰을 찾을 수 없습니다!")
    
    try:
        unk_id = sp.unk_id()
        unk_piece = sp.id_to_piece(unk_id)
        print(f"- <unk> 토큰 ID: {unk_id}, 토큰: {unk_piece}")
    except:
        print("- <unk> 토큰 ID를 가져오는 데 문제가 발생했습니다")
    
    print(f"- pad_id: {sp.pad_id()}")
    print(f"- bos_id: {sp.bos_id()}")
    print(f"- eos_id: {sp.eos_id()}")
    
    print("\n처음 10개 토큰:")
    for i in range(min(10, sp.get_piece_size())):
        print(f"  {i}: '{sp.id_to_piece(i)}'")
    
    print("\n샘플 텍스트 인코딩 테스트:")
    for text in text_samples:
        print(f"\n원본 텍스트: {text}")
        
        # 텍스트를 토큰 ID로 인코딩
        ids = sp.encode(text, out_type=int)
        print(f"인코딩된 ID: {ids}")
        
        # 인코딩된 ID를 다시 텍스트로 디코딩
        decoded = sp.decode(ids)
        print(f"디코딩된 텍스트: {decoded}")
        
        # 토큰 단위로 출력
        pieces = sp.encode(text, out_type=str)
        print(f"토큰 리스트: {pieces}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SentencePiece 토크나이저 검증')
    parser.add_argument('--model', required=True, help='SentencePiece 모델 경로')
    args = parser.parse_args()
    
    # 영어 샘플 텍스트로 변경
    samples = [
        "I WANT TO BE DOING SOMETHING ON MY OWN ACCOUNT",
        "OH EVER SO MUCH ONLY HE SEEMS KIND OF STAID AND SCHOOL TEACHERY",
        "FRANK READ ENGLISH SLOWLY AND THE MORE HE READ ABOUT THIS DIVORCE CASE THE ANGRIER HE GREW"
    ]
    
    validate_tokenizer(args.model, samples)