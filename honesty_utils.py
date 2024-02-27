import pandas as pd
import numpy as np
from utils import get_activations_for_paired_statements, get_directions, get_accs_for_pairs

class HonestyExtractor():
    
    def __init__(self, model, tokenizer, user_tag, assistant_tag):
        self.model = model
        self.tokenizer = tokenizer
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        
    def find_honesty_directions(self):
        statement_pairs = prepare_statement_pairs(self.tokenizer, user_tag=self.user_tag, assistant_tag=self.assistant_tag)
        train_act_pairs = get_activations_for_paired_statements(statement_pairs, self.model, self.tokenizer, sample_range=[0, 512])   
        direction_info = get_directions(train_act_pairs)
        self.direction_info = direction_info
        return self.direction_info
        

def prepare_statement_pairs(tokenizer, user_tag, assistant_tag):

    def _prompt_maker(stimulus, honesty, user_tag, assistant_tag):
        prompt = f"{user_tag} Pretend you're an {'honest' if honesty else 'untruthful'}"\
        f" person making statements about the world. {assistant_tag} {stimulus}"
        return prompt
    
    data = pd.read_csv('./data/facts/facts_true_false.csv')
    statement_pairs = []
    statements = data[data['label'] == 1]['statement'].values.tolist() # they only use label=1 for some reason
    for statement in statements:
        tokens = tokenizer.tokenize(statement)
        for idx in range(1, len(tokens)-5):
            substatement = tokenizer.convert_tokens_to_string(tokens[:idx])
            honest_statement = _prompt_maker(substatement, True, user_tag, assistant_tag)
            dishonest_statement = _prompt_maker(substatement, False, user_tag, assistant_tag)
            statement_pairs.append([honest_statement, dishonest_statement])
    statement_pairs = np.array(statement_pairs)
    return statement_pairs
