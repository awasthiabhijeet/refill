import json
import argparse 
import random 
from pathlib import Path

random.seed(1618)

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=int, help='Threshold for filtering', default=10) 
parser.add_argument('--root_dir', type=str, help='Root directory to grouped splits') 
args = parser.parse_args()

dev_frac = {0.3: 1., 0.5: 0.5, 0.7: 0.3}

for frac in [0.3, 0.5, 0.7]:
    root_dir = Path(f'{args.root_dir}/{frac}')
    for group in range(1, 5):
        with open(root_dir / f'group_{group}/group_{group}_entries_list.json') as f:
            entries = json.load(f)
        with open(root_dir / f'group_{group}/group_{group}_train_gazp_output.json') as f:
            all_train = json.load(f)
        proc = [{'exec': 1 if d['exec'] else 0, 'partial': sum([d['partial'][k]['f1'] for k in d['partial']])} for d in entries]
        proc = [x['exec'] * 10 + x['partial'] for x in proc]
        
        final = []
        for i, val in enumerate(proc):
            if val >= args.threshold:
                final.append(all_train[i])

        num_dev = int(dev_frac[frac] * len(final)) + 1 
        num_dev = min(num_dev, len(final))

        dev = random.sample(final, num_dev)

        with open(root_dir / f'group_{group}/finetune_train.json', 'w') as f:
            json.dump(final, f, indent=4, sort_keys=True)

        with open(root_dir / f'group_{group}/finetune_dev.json', 'w') as f:
            json.dump(dev, f, indent=4, sort_keys=True)
