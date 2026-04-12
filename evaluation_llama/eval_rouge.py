import json
import argparse
import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer

def load_cnndm_references(seed, data_num=10):
    # Using the exact same dataset loading and shuffling as in eval.py
    data = load_dataset('cnn_dailymail', name='3.0.0', split='test').shuffle(seed=seed).select(range(data_num))
    references = []
    for item in data:
        # CNN/DM reference summary
        references.append(item['highlights'].replace('\n', ' '))
    return references

def evaluate_rouge(answer_file, seed, data_num):
    print(f"Loading references from CNN/DM (seed={seed}, data_num={data_num})...")
    references = load_cnndm_references(seed, data_num)
    
    print(f"Loading predictions from {answer_file}...")
    predictions = []
    with open(answer_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            # choices is a list containing a dictionary with "turns"
            # It looks like: {"choices": [{"turns": "output text", ...}], ...}
            if "choices" in data:
                pred = data["choices"][0]["turns"]
                predictions.append(pred.replace('\n', ' '))
            elif "Mean accepted tokens" in data:
                # This is the statistics footer, ignore
                continue
                
    if len(predictions) == 0:
        print("No predictions found in the file.")
        return
        
    if len(predictions) != len(references):
        print(f"Warning: Number of predictions ({len(predictions)}) does not match references ({len(references)}). Evaluating first {min(len(predictions), len(references))} sequences.")
    
    eval_len = min(len(predictions), len(references))
    predictions = predictions[:eval_len]
    references = references[:eval_len]
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    r1_fmeasures = []
    r2_fmeasures = []
    rL_fmeasures = []
    
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        r1_fmeasures.append(scores['rouge1'].fmeasure)
        r2_fmeasures.append(scores['rouge2'].fmeasure)
        rL_fmeasures.append(scores['rougeL'].fmeasure)
        
    print(f"\n--- ROUGE Scores ({eval_len} samples) ---")
    print(f"ROUGE-1: {np.mean(r1_fmeasures):.4f}")
    print(f"ROUGE-2: {np.mean(r2_fmeasures):.4f}")
    print(f"ROUGE-L: {np.mean(rL_fmeasures):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated JSONL answers against CNN/DM with ROUGE")
    parser.add_argument("--answer-file", type=str, required=True, help="Path to the JSONL output file")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed used in generation (default: 2024)")
    parser.add_argument("--data-num", type=int, default=10, help="Number of samples generated (default: 10)")
    
    args = parser.parse_args()
    evaluate_rouge(args.answer_file, args.seed, args.data_num)
