import os
import argparse


def txt_generation(scp_path):
    txts = []
    
    with open(scp_path, 'r') as f:
        scp_lines = f.readlines()
        for line in scp_lines:
            _, transcription = line.split('|')
            txts.append(transcription)
    f.close()
    
    txt_path = './input.txt'
    with open(txt_path, 'w') as txt:
        for t in txts:
            txt.write(t)
    txt.close()

def main():
    parser = argparse.ArgumentParser(description='generating input txt file for spm training')
    parser.add_argument('--scp_path')
    args = parser.parse_args()
    txt_generation(args.scp_path)


if __name__=='__main__':
    main()