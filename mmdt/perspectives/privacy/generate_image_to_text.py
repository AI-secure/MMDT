
import argparse
from location.generate_response_location_privacy import *
from pii.generate_response_pii import generate_pii_response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use for evaluation')
    parser.add_argument('--scenario', type=str, default="location", help='Specify the scenario')
    parser.add_argument('--task', type=str, default="Pri-SV-with-text", help='Specify the task to execute')
    args = parser.parse_args()
    
    if(args.scenario=="location"):
        if args.task in ["Pri-4Loc-SV-with-text", "Pri-4Loc-SV-without-text"]:
            generate_response_4img(args.model_id, args.task)
        elif args.task in ["Pri-SV-with-text","Pri-SV-without-text"]:
            generate_response_1img(args.model_id, args.task)
    elif (args.scenario == "pii"):
        generate_pii_response(args.model_id)
