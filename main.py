import csv
from logs.filelister import filelister
from online_predictor import OnlineActivityPredictor
import torch.cuda
import random
import numpy as np
from sklearn.metrics import f1_score	
import time
from collections import defaultdict, deque
import re
from cust_parser import parse_args

# Set random seed for reproducibility
SEED = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

BATCH_SIZE = 32

event_count=0
current_conc_idx=0
concept_history={}

def normalise(event):
    return event.strip().lower()


def process_batch(batch_rows, concept_drift, taskies,task_oracle=None, final_outcome_flag=None, noisy=False, on_concept_drift=None):
    global total_predictions, correct_predictions, event_count, current_conc_idx, concept_history
    
    for row in batch_rows:

        if task_oracle is not None:
            if event_count in task_oracle:
                final_outcome_flag = not final_outcome_flag


        if event_count in concept_drift:
            print(f"\n--- Concept Drift Detected at event sampleunprinted {event_count} ---")
            prev_conc=current_conc_idx
            current_conc_idx=taskies[concept_drift.index(event_count)]

            if predictor.predictions:
                concept_f1=f1_score(predictor.targets, predictor.predictions,average='weighted')
                concept_acc= sum([1 for p,t in zip(predictor.predictions,predictor.targets)if p==t])/len(predictor.predictions)
                
                print(f"Concept {prev_conc} to {current_conc_idx} | {prev_conc} stats are F1: {concept_f1:.4f} | Acc: {concept_acc:.4f}")

                concept_history.setdefault(prev_conc,{})
                concept_history[prev_conc]['last_f1']=concept_f1
                concept_history[prev_conc]['last_acc']=concept_acc
                
                #if new concept has already happened in the past
                if current_conc_idx in concept_history:
                    prev_f1 = concept_history[current_conc_idx].get('last_f1')
                    prev_acc = concept_history[current_conc_idx].get('last_acc')
                    print(f"Previous stats for new concept {current_conc_idx} | F1: {prev_f1:.4f} | Acc: {prev_acc:.4f}")

                concept_history.setdefault(current_conc_idx,{})
                concept_history[current_conc_idx]['first_f1']=None
                concept_history[current_conc_idx]['first_acc']=None

                predictor.predictions = []
                predictor.targets = []

                #record for task specific forgetting (acc and f1 based)
                if 'concept_stats' not in globals():
                    global concept_stats
                    concept_stats = {}
                concept_stats[prev_conc] = (concept_f1, concept_acc)

                if current_conc_idx in concept_stats:
                    prev_f1, prev_acc = concept_stats[current_conc_idx]
                    print(f"Previous Concept {current_conc_idx} | F1: {prev_f1:.4f} | Acc: {prev_acc:.4f}")
                else:
                    print(f"New Concept {current_conc_idx} | No previous stats available.")
        
                if on_concept_drift is not None:
                    on_concept_drift(prev_conc, concept_acc, concept_f1)
        case = row['case']
        # Find the final row with the same case value as the current
        final_row_idx = len(batch_rows) - 1
        current_idx = batch_rows.index(row)
        while final_row_idx > current_idx and batch_rows[final_row_idx]['case'] != case:
            final_row_idx -= 1
        final_row = batch_rows[final_row_idx]
        if not noisy:
            current_activity = row['event']
            final_activity = final_row['event']
        else:
            current_activity = row['true_label']
            final_activity = final_row['true_label']
        previous_activities = case_buffer[case]

        if previous_activities:

            predicted = predictor.predict_and_update(case, previous_activities, current_activity, final_activity, concept_idx=current_conc_idx, log=True, final_outcome_flag=final_outcome_flag)

            total_predictions += 1
            if predicted == current_activity:
                correct_predictions += 1

            # print(f"[Case {case}] Predicted: {predicted} | Actual: {current_activity} | Time: {exec_time:.4f}s")

        case_buffer[case].append(current_activity)
        event_count += 1
    return final_outcome_flag
args = parse_args()

clean_BPIC_datasets = filelister("local_datasets/BPIC", with_path=1, file_extension=".csv", avoid_files="oracle.csv", startswith="BPIC")
noisy_BPIC_datasets = filelister("local_datasets/BPIC", with_path=1, file_extension=".csv", avoid_files="oracle.csv", startswith="Sym")
tams_datasets = filelister("local_datasets/Tams", with_path=1, file_extension=".csv", avoid_files="oracle.csv")


datasets=noisy_BPIC_datasets

for ds in datasets:
    event_count = 0
    current_conc_idx = 0
    concept_history = {}
    

    #setup the predictor
    case_buffer = defaultdict(list)  #dictionary with the case prefix
    oracle_path = ds.replace('.csv', '_oracle.csv')
    pattern = re.compile(r"SymN(?:20|40|60)_")
    
    if ds.startswith("local_datasets/Tams/Sym") or ds.startswith("local_datasets/BPIC/Sym"):
        oracle_path = re.sub(pattern, '', oracle_path)
        noisy = True
    else:
        noisy = False

  
    with open(oracle_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f) 
        concept_drift = [int(x) for x in next(reader)]
    if '_Rec' in ds:
        taskies = [1,2,3,2,3,1,2,3,2,3,1,2,3,2,3,1,2,3,2,3,1,2,3,2,3,1,2,3,2,3,1,2,3,2,3,1]
    elif '_Imb' in ds:
        taskies = [1,2,3,4,3,1,3,2,1,2,1]
    elif '_Rng' in ds:
        taskies = [1,2,3,4,5,6,1,3,6,4,2,6,3,1,2,5,6,4,5,1,4,2,6,3]
    else:
        raise ValueError(f"Dataset {ds} does not match expected shape")
    
    num_concepts = max(taskies)+1

    proportion_drift = [0.25, 0.5, 0.75] #proportions through the dataset where task changes
    task_oracle={
        concept_drift[round(p*len(concept_drift)-1)] for p in proportion_drift
        #set guarantees O(1) lookup
    }
    final_outcome_flag = False
    
    activity_labels=set()
    with open("local_datasets/warmup_set.csv",'r',encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            activity_labels.add(normalise(row['event']))

    with open(ds,'r', encoding ='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            activity_labels.add(normalise(row['event']))
    
    num_activities = len(activity_labels)

    predictor = OnlineActivityPredictor(lr=args.lr, pd_balance=args.pd_balance, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, 
                                        num_heads=args.num_heads, num_concepts=num_concepts, 
                                        num_activities=num_activities, use_gru=args.use_gru, 
                                        use_pdScore=args.use_pd, use_mhsa=args.use_mhsa, 
                                        use_G_prompt=args.use_G_prompt, use_E_prompt=args.use_E_prompt, 
                                        use_T_prompt=args.use_T_prompt)

    # #warmupSlice
    slice_size = args.slice_size
    epochs = args.epochs
    if args.warmup_slice:
        print("Using warmup slice for training.")
        for epoch in range(epochs):
            with open(ds,'r',encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = []
                for i, row in enumerate(reader):
                    if i >= slice_size:
                        break
                    rows.append(row)
                case_buffer = defaultdict(list)
                for i,row in enumerate(rows):
                    if i>= slice_size:
                        break
                    case = row['case']
                    final_row_idx = len(rows) - 1
                    while final_row_idx > i and rows[final_row_idx]['case'] != case:
                        final_row_idx -= 1
                    final_row = rows[final_row_idx]
                    if not noisy:
                        current_activity = row['event']
                        final_activity = final_row['event']
                    else:
                        current_activity = row['true_label']
                        final_activity = final_row['true_label']
                    previous_activities = case_buffer[case]
                    if previous_activities:
                        predictor.predict_and_update(case, previous_activities, current_activity, final_activity, concept_idx=current_conc_idx, log=False, final_outcome_flag=final_outcome_flag)
                    case_buffer[case].append(current_activity)
        print("Warmup slice training completed.")

###WARMUP SET WAS EMPIRICALLY PROVEN WORSE, AND IT IS NOT USED ANYMORE
    # elif args.warmup_set:
    #     print("Using warmup set for training.")
        
    #     if not noisy:
    #         print("Using local warmup set for training, no noise.")
    #         warmup_set = "local_datasets/warmup_set.csv"
    #     else:
    #         print("Using noisy warmup set for training.")
    #         if ds.startswith("local_datasets/Tams/SymN20_"):
    #             warmup_set = "local_datasets/SymN20_warmup_set.csv"
    #         elif ds.startswith("local_datasets/Tams/SymN40_"):
    #             warmup_set = "local_datasets/SymN40_warmup_set.csv"
    #         elif ds.startswith("local_datasets/Tams/SymN60_"):
    #             warmup_set = "local_datasets/SymN60_warmup_set.csv"
                        
    #     if not warmup_set:
    #         raise ValueError("No warmup set provided. Please provide a valid warmup set file.")
        
    #     for epoch in range(20):
    #         with open(warmup_set,'r',encoding='utf-8-sig') as f:
    #             reader = csv.DictReader(f)
    #             case_buffer = defaultdict(list)
    #             for i,row in enumerate(reader):
    #                 case = row['case']
    #                 current_activity = row['event']
    #                 previous_activities = case_buffer[case]
    #                 if previous_activities:
    #                     predictor.predict_and_update(case, previous_activities, current_activity, concept_idx=current_conc_idx,log=False)
    #                 case_buffer[case].append(current_activity) 
    else:
        raise ValueError("no warmup strategy selected")


    # Accuracy counters
    total_predictions = 0
    correct_predictions = 0
    case_buffer = defaultdict(list)
    predictor.predictions = []
    predictor.targets = []
    predictor.predictions_count = 0


    with open(ds, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        batch = []
        start_time = time.time()
        rows=list(reader)
        for i,row in enumerate(rows):
            batch.append(row)
            if i == len(rows)-1 or (row['case'] != rows[i+1]['case'] and len(batch) >= BATCH_SIZE):
                final_outcome_flag=process_batch(batch,concept_drift, taskies,task_oracle, final_outcome_flag,noisy)
                batch = []

    # Final predictions without targets
    for case, last_activity in case_buffer.items():
        predictor.predict_and_update(case, last_activity, next_activity=None, concept_idx=current_conc_idx,log=True, final_outcome_flag=final_outcome_flag)
    exec_time = time.time()-start_time
    # Final accuracy report
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n Dataset: {ds}\n✅ Total Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        print("⏱️ Average prediction time: {:.4f}s".format(exec_time / total_predictions))
    else:
        print("⚠️ No predictions were made.")
    total_predictions = 0
    correct_predictions = 0
    predictor.predictions_count = 0
