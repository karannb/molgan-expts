import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem
from collections import OrderedDict
from deepchem.models.torch_models import BasicMolGANModel as MolGAN

from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix

import torch
from torch.nn.functional import one_hot

best_unique = -1
best_valid = -1

name2dset = {
    "qm7": dc.molnet.load_qm7,
    "qm9": dc.molnet.load_qm9,
    "bbbp": dc.molnet.load_bbbp,
    "lipo": dc.molnet.load_lipo,
    "pdb": dc.molnet.load_pdbbind,
}


def iterbatches(epochs, dataset, feat, gan, early_stop=False):
    
    global best_unique
    global best_valid
    
    for i in range(epochs):
        for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            flattened_adjacency = torch.from_numpy(batch[0]).view(-1).to(dtype=torch.int64) # flatten the input because torch.nn.functional.one_hot only works with 1D inputs
            invalid_mask = (flattened_adjacency < 0) | (flattened_adjacency >= gan.edges) # edge type cannot be negative or >= gan.edges, these entries are invalid
            clamped_adjacency = torch.clamp(flattened_adjacency, 0, gan.edges-1) # clamp the input so it can be fed to the one_hot function
            adjacency_tensor = one_hot(clamped_adjacency, num_classes=gan.edges) # actual one_hot
            adjacency_tensor[invalid_mask] = torch.zeros(gan.edges, dtype=torch.long) # make the invalid entries, a vector of zeros
            adjacency_tensor = adjacency_tensor.view(*batch[0].shape, -1) # reshape to original shape.

            flattened_node = torch.from_numpy(batch[1]).view(-1).to(dtype=torch.int64)
            invalid_mask = (flattened_node < 0) | (flattened_node >= gan.nodes)
            clamped_node = torch.clamp(flattened_node, 0, gan.nodes-1)
            node_tensor = one_hot(clamped_node, num_classes=gan.nodes)
            node_tensor[invalid_mask] = torch.zeros(gan.nodes, dtype=torch.long)
            node_tensor = node_tensor.view(*batch[1].shape, -1)

            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]: node_tensor}
            
        if (i+1) % 10 == 0 and early_stop:
            generated_data = gan.predict_gan_generator(6400)
            nmols = feat.defeaturize(generated_data)
            print("{} molecules generated".format(len(nmols)))

            nmols = list(filter(lambda x: x is not None, nmols))
            # currently training is unstable so 0 is a common outcome
            print ("{} valid molecules".format(len(nmols)))

            nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
            nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
            nmols_viz = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]
            print ("{} unique valid molecules".format(len(nmols_viz)))
            
            if len(nmols_viz) > best_unique:
                best_unique = len(nmols_viz)
                best_valid = len(nmols)
            
            if len(nmols_viz) < 128:
                print(f"Exiting due to early stop at epoch {i}")
                break


def parse():
    
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["normal", "gumbel", "straight"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["qm7", "qm9", "bbbp", "lipo", "pdb"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--generator_steps", type=float, default=0.2)
    args = parser.parse_args()
    
    if args.mode == "normal":
        args.mode = ""
    
    return args


def main(args):

    dropout = args.dropout # training unstable for 0.1, haven't tested 0.25
    generator_steps = args.generator_steps
    mode = args.mode # can set to "" or "gumbel" or "straight"

    # Prepare data
    print(f"Preparing data for {args.dataset}...")
    dset = name2dset[args.dataset]
    _, datasets, _ = dset()
    df = pd.DataFrame(data={'smiles': datasets[0].ids})

    print(f"Featurizing data for {args.dataset}...")
    feat = dc.deepchem.feat.MolGanFeaturizer()
    data = df
    smiles = data['smiles'].values
    filtered_smiles = [x for x in smiles if Chem.MolFromSmiles(x).GetNumAtoms() < 9]
    features = feat.featurize(filtered_smiles)
    # remove non-GraphMatrix features
    features = [x for x in features if type(x) == GraphMatrix]
    print(f"Found {len(filtered_smiles)} molecules with < 9 atoms.")
    dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features],
                                   [x.node_features for x in features])

    out_csv = {"seed": [],
               "validity": [],
               "uniqueness": []}
    for cur_seed in range(120, 130):
        np.random.seed(cur_seed)
        torch.manual_seed(cur_seed)

        # train model
        gan = MolGAN(edges=4, 
                     vertices=9, 
                     embedding_dim=32, 
                     mode=mode, 
                     learning_rate=0.0001, 
                     dropout_rate=dropout, 
                     batch_size=32)
        gan.fit_gan(iterbatches(30, dataset, feat, gan), 
                    generator_steps=generator_steps, 
                    checkpoint_interval=2500)

        print(f"Training for seed {cur_seed} complete.")
        print(f"Generating molecules for seed {cur_seed}...")
        generated_data = gan.predict_gan_generator(6400)
        nmols = feat.defeaturize(generated_data)
        print("{} molecules generated".format(len(nmols)))

        nmols = list(filter(lambda x: x is not None, nmols))
        # currently training is unstable so 0 is a common outcome
        print ("{} valid molecules".format(len(nmols)))

        nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
        nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
        nmols_viz = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]
        print ("{} unique valid molecules".format(len(nmols_viz)))
        
        unique = len(nmols_viz)
        valid = len(nmols)

        out_csv["seed"].append(cur_seed)
        out_csv["uniqueness"].append(unique)
        out_csv["validity"].append(valid)

    df = pd.DataFrame(out_csv, index=None)
    df.to_csv(f"mode_{args.mode}_dropout_{args.dropout}_gen_steps_{args.generator_steps}_{args.name}_results.csv")
    
    return


if __name__ == '__main__':
    
    args = parse()
    main(args)