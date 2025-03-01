import argparse
from location.calculate_acc_rej import calculate_acc_rej
from pii.calculate_metrics import calculate_pii

def evaluate(kwargs):
    model_id, scenario, task = kwargs.model_id, kwargs.scenario, kwargs.task
    
    if(scenario=="location"):
        calculate_acc_rej()# analyse all the response file

    elif (scenario=="pii"):
        calculate_pii()
