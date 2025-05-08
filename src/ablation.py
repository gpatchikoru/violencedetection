import argparse
from train import main as train_main
from evaluate_model import main as eval_main

def run_ablation(h5, skips, temporals):
    results = {}
    for skip in skips:
        print(f"Running skip={skip}")
        for temp in temporals:
            print(f"  Temporal head: {temp}")
            train_main(['--h5', h5, '--epochs','10','--skip',str(skip),'--temporal',temp])
            eval_main(['--model','checkpoints/last.pth','--h5',h5])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--skips', nargs='+', type=int)
    p.add_argument('--temporals', nargs='+', type=str)
    args = p.parse_args()
    run_ablation(args.h5, args.skips, args.temporals)

if __name__=='__main__':
    main()