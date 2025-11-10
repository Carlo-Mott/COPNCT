import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import defaultdict
from mhsa import MultiHeadSelfAttention
from g_prompt import GPrompt
from e_prompt import EPrompt
from t_prompt import TPrompt
from sklearn.metrics import f1_score
from purity_diversity import PurityDiversity

# Set random seed for reproducibility
SEED = 1
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

class NextActivityPredictorModel(nn.Module):
    def __init__(self, *, embedding_dim=32, hidden_dim=64, num_heads=4,
                g_prompt_len=5, e_prompt_len=5, num_concepts=-1, num_activities=-1,
                use_gru=0, use_mhsa=1, use_G_prompt=1, use_T_prompt=0, use_E_prompt=1,
                use_pdScore=1):
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        self.device = device

        #component flags
        self.use_gru = use_gru
        self.use_G_prompt = use_G_prompt
        self.use_T_prompt = use_T_prompt
        self.use_E_prompt = use_E_prompt
        self.use_mhsa = use_mhsa
        self.use_pdScore = use_pdScore

        #Embedding and GRU
        self.embedding = nn.Embedding(num_activities, embedding_dim)
        if self.use_gru:
            self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            self.embedding_projection = nn.Linear(embedding_dim, hidden_dim)  
        
        #MHSAs
        if self.use_mhsa:
            self.attention1 = MultiHeadSelfAttention(hidden_dim, embedding_dim, num_heads)
            self.attention2 = MultiHeadSelfAttention(hidden_dim, embedding_dim, num_heads)
            self.attention3 = MultiHeadSelfAttention(hidden_dim, embedding_dim, num_heads)
        
        #Prompts (General, Expert, Prediction_task)
        head_dim = hidden_dim // num_heads
        if self.use_G_prompt:                            #uniform, normal, xavier, zeros.-- xavier gives best results in CNAPwP
            self.g_prompt = GPrompt(num_heads, hidden_dim, g_prompt_len, prompt_init='xavier') 

        if self.use_T_prompt:
            self.t_prompt = TPrompt(num_heads, hidden_dim, prompt_init='xavier')  
        
        if self.use_E_prompt:
            self.e_prompt = EPrompt(num_concepts, num_heads, hidden_dim, prompt_init='xavier')  

        #linears and dropout
        self.linear1= nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))	
        self.linear2= nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.linear3= nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.output_layer = nn.Linear(hidden_dim, num_activities)



    def forward(self, input_seq, hidden_state=None, concept_idx=None, final_outcome_flag=None):
        x = self.embedding(input_seq)  
        
        if self.use_gru:
            out, hidden = self.gru(x, hidden_state) 
        else:
            out = self.embedding_projection(x)
            hidden=None
        
        #MHSA and linear layers
        if self.use_mhsa:
            #G_Prompt: General Prompt
            if self.use_G_prompt:    
                out = self.attention1(out, prompt=self.g_prompt.get_prompt())
                out = self.linear1(out)
            else:
                out = self.attention1(out)
                out = self.linear1(out)
            #T_Prompt: Task Prompt
            if self.use_T_prompt:
                prompt_idx = 1 if final_outcome_flag else 0
                out = self.attention2(out, prompt=self.t_prompt.get_prompt(prompt_idx))
                out = self.linear2(out)
            else:
                out = self.attention2(out)
                out = self.linear2(out)
            #E_Prompt: Expert Prompt
            if self.use_E_prompt:
                k,v=self.e_prompt.get_prompt(concept_idx)
                out = self.attention3(out, prompt=(k, v))
                out = self.linear3(out)
            else:
                out = self.attention3(out)
                out = self.linear3(out)
        else:
            out = self.linear1(out)

        logits = self.output_layer(out[:, -1])  #Only last timestep output
        # pooled = out.mean(dim=1)  #Mean pooling over time
        # logits = self.output_layer(pooled)  #Final output layer on pooled representation
        return logits, hidden

class OnlineActivityPredictor:
    def __init__(self, *, lr=0.0001, pd_balance=0.5, embedding_dim=32, hidden_dim=64, num_heads=4, log_interval=100, num_concepts=-1, num_activities=-1, use_gru=0, use_mhsa=1, use_G_prompt=1, use_T_prompt=0, use_E_prompt=1, use_pdScore=1, max_len=64):
        
        if num_concepts == -1:
            raise ValueError("num_concepts is -1, meaning it was not passed to the constructor")
        if num_activities == -1:
            raise ValueError("num_activities is -1, meaning it was not passed to the constructor")
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads=num_heads
        self.num_concepts = num_concepts
        self.num_activities = num_activities
        self.log_interval = log_interval
        self.max_len=max_len
        self.dynamic_weights = PurityDiversity(balance=pd_balance)  # Memory manager for purity-diversity
        
        self.activity_to_index = {}  #maps activity_id to index
        self.index_to_activity = {}  #reverse mapping

        self.model = NextActivityPredictorModel(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_heads=num_heads,num_concepts=num_concepts, num_activities=num_activities,use_gru=use_gru, use_mhsa=use_mhsa, use_G_prompt=use_G_prompt, use_T_prompt=use_T_prompt, use_E_prompt=use_E_prompt, use_pdScore=use_pdScore).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.use_pdScore = self.model.use_pdScore
        self.case_states = defaultdict(lambda: None)  # case_id -> GRU hidden state

        self.predictions = []
        self.targets = []
        self.predictions_count = 0

    def _ensure_activity_known(self, activity_id):
        if activity_id not in self.activity_to_index:
            new_index = len(self.activity_to_index)
            if new_index > self.num_activities:
                raise ValueError(f"Exceeded maximum number of activities{new_index} >= {self.num_concepts}")
            self.activity_to_index[activity_id] = new_index
            self.index_to_activity[new_index] = activity_id

    def log_f1(self, concept_idx=None):
        global concept_history
        if self.predictions_count % self.log_interval == 0:
            f1 = f1_score(self.targets, self.predictions,average='weighted')
            acc = sum([1 for p,t in zip(self.predictions, self.targets) if p==t]) / len(self.targets)
            print(f"F1 at {self.predictions_count}: {f1}")
            print(f"Acc at {self.predictions_count}: {acc}")

            #task specific forgetting 
            if concept_idx is not None and 'concept_history' in globals():
                concept_history = globals()['concept_history']
                ch = concept_history.setdefault(concept_idx,{})
                
                if 'last_f1' in ch and 'last_acc' in ch:
                    forg_f1 = ch['last_f1'] - f1
                    forg_acc = ch['last_acc'] - acc
                    print(f"Task specific forgetting: F1: {forg_f1}, Acc: {forg_acc}")
                else:
                    ch['last_f1'] = f1
                    ch['last_acc'] = acc
                    print(f"No previous F1/Acc found for this concept ({concept_idx}), storing current values.")

            self.predictions = []
            self.targets = []
            return f1

    def predict_and_update(self, case_id, case_prefix, next_activity=None, final_activity=None, concept_idx=None, log=False, final_outcome_flag=None):

        # Ensure current and next activities are known
        for activity in case_prefix:
            # if self.max_len is not None and len(case_prefix) > self.max_len:
            #     case_prefix = case_prefix[-self.max_len:]
            self._ensure_activity_known(activity)
        if next_activity is not None:
            self._ensure_activity_known(next_activity)
        if final_activity is not None:
            self._ensure_activity_known(final_activity)
        self.log=log

        # Convert to index
        indices = [self.activity_to_index[act] for act in case_prefix]
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
        hidden_state = self.case_states[case_id]

        # Prediction
        logits, new_hidden = self.model(input_tensor, hidden_state, concept_idx=concept_idx, final_outcome_flag=final_outcome_flag)
        
        # Store updated state
        if new_hidden is not None:
            self.case_states[case_id] = new_hidden.detach()

        # Online training
        target = final_activity if final_outcome_flag else next_activity
        if target is not None:
            target_idx = self.activity_to_index[target]
            target_tensor = torch.tensor([target_idx], dtype=torch.long).to(self.device)
            loss = self.criterion(logits, target_tensor)
            
            known_indices = list(self.index_to_activity.keys())
            max_index=logits.shape[1]
            for idx in known_indices:
                if idx >= max_index:
                    raise IndexError(f"Index {idx} is out of bounds for logits with shape {logits.shape}")
            filtered_logits = logits[:,known_indices]

            probs = torch.softmax(filtered_logits, dim=1)
            entropy = -torch.sum(probs*torch.log(probs + 1e-8), dim=1).item()

            # Calculate pd-informed dynamic weights
            if self.use_pdScore:
                norm_score=1-self.dynamic_weights.update({'loss':loss.item(),'uncertainty':entropy,'concept':concept_idx}) #swapping to higher is better
                weighted_loss = loss * norm_score
            else:
                weighted_loss = loss
            
            self.optimizer.zero_grad()

            weighted_loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            known_indices = list(self.index_to_activity.keys())
            max_index=logits.shape[1]
            
            for idx in known_indices:
                if idx >= max_index:
                    raise IndexError(f"Index {idx} is out of bounds for logits with shape {logits.shape}")
            
            filtered_logits = logits[:,known_indices]
            
            selected_idx = torch.argmax(filtered_logits, dim=1).item()
            predicted_idx= known_indices[selected_idx]
            predicted_activity = self.index_to_activity[predicted_idx]

        if next_activity is not None:
            self.predictions.append(predicted_idx)
            self.targets.append(target_idx)
            self.predictions_count += 1
            if self.log:
                self.log_f1(concept_idx=concept_idx)

            
           
        return predicted_activity
