
import os
from tqdm import tqdm
import argparse
from PIL import Image
from location.calculate_acc_rej import calculate_acc_rej


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for evaluation')
    parser.add_argument('--scenario', type=str, default="location", help='Specify the scenario')
    parser.add_argument('--task', type=str, default="Pri-SV-with-text", help='Specify the task to execute')
    args = parser.parse_args()
    
    if(args.scenario=="location"):
        calculate_acc_rej()# analyse all the response file
        
    #else PII
