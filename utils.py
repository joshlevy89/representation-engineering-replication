import torch
from collections import defaultdict
from sklearn.decomposition import PCA
import numpy as np


def get_activations_for_paired_statements(statement_pairs, model, tokenizer, sample_range, read_token=-1, batch_size=16, device='cuda:0'):
    layer_to_act_pairs = defaultdict(list)
    for i in range(sample_range[0], sample_range[1], batch_size):
        pairs = statement_pairs[i:i+batch_size]
        statements = pairs.reshape(-1)
        model_inputs = tokenizer(list(statements), padding=True, return_tensors='pt').to(device)
        with torch.no_grad():
            hiddens = model(**model_inputs, output_hidden_states=True)
        for layer in range(model.config.num_hidden_layers):
            act_pairs = hiddens['hidden_states'][layer+1][:, read_token, :].view(batch_size, 2, -1)
            layer_to_act_pairs[layer].extend(act_pairs)
    
    for key in layer_to_act_pairs:
        layer_to_act_pairs[key] = torch.stack(layer_to_act_pairs[key])

    return layer_to_act_pairs


def get_directions(train_acts, device='cuda:0'):
    directions = {}
    signs = {}
    mean_diffs = {}
    direction_info = defaultdict(dict)
    for layer in train_acts:
        act_pairs = train_acts[layer]
        shuffled_pairs = [] # shuffling train labels before pca useful for some reason 
        for pair in act_pairs:
            pair = pair[torch.randperm(2)]
            shuffled_pairs.append(pair)
        shuffled_pairs = torch.stack(shuffled_pairs)
        diffs = shuffled_pairs[:, 0, :] - shuffled_pairs[:, 1, :] 
        mean_diffs[layer] = torch.mean(diffs, axis=0)
        centered_diffs = diffs - mean_diffs[layer] # is centering necessary?
        pca = PCA(n_components=1)
        pca.fit(centered_diffs.detach().cpu())
        directions[layer] = torch.tensor(pca.components_[0], dtype=torch.float16).to(device)
        
        # get signs
        projections = do_projections(train_acts[layer], directions[layer], 1, mean_diffs[layer])
        acc = np.mean([(pi > pj).item() for (pi, pj) in projections])
        sign = -1 if acc < .5 else 1
        signs[layer] = sign
    direction_info['directions'] = directions
    direction_info['signs'] = signs
    direction_info['mean_diffs'] = mean_diffs
    return direction_info


def do_projections(acts, direction, sign, mean_diff, center=True, layer=None):
    if center:
        acts = (acts - mean_diff).clone()
    projections = sign * acts @ direction / direction.norm() # i don't think this projection is exactly right
    return projections


def get_accs_for_pairs(test_acts, direction_info):
    directions = direction_info['directions']
    signs = direction_info['signs']
    mean_diffs = direction_info['mean_diffs']
    accs = []
    for layer in test_acts:
        projections = do_projections(test_acts[layer], directions[layer], signs[layer], mean_diffs[layer])
        acc = np.mean([(pi > pj).item() for (pi, pj) in projections])
        accs.append(acc)
    return accs