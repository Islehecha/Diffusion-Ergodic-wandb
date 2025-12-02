#!/usr/bin/env python3
"""Update config from dataset_analysis/config_update.yaml"""
import os
import yaml
from datetime import datetime

def main():
    stats_path = 'dataset_analysis/config_update.yaml'
    config_path = 'diffusion_ergodic/config/config_ergodic.yaml'
    if not os.path.exists(stats_path):
        print('Missing', stats_path)
        return 1
    with open(stats_path, 'r') as f:
        new_norm = yaml.safe_load(f)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # backup
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.rename(config_path, config_path + '.bak_' + ts)
    # update
    config['normalizer'] = new_norm.get('normalizer', config.get('normalizer', {}))
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print('Config updated and backed up as', config_path + '.bak_' + ts)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

