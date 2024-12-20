import os
import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem, rdBase
from collections import OrderedDict
from evaluate import MolecularMetrics
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
    "ppb": dc.molnet.load_ppb,
}


def make_one_hot(x: torch.Tensor, num_classes: int):
    """
    One hot encode a tensor
    """
    flat_tensor = x.view(-1).to(dtype=torch.int64) # flatten the input because torch.nn.functional.one_hot only works with 1D inputs
    invalid_values = (flat_tensor < 0) | (flat_tensor >= num_classes) # x cannot be negative or >= num_classes, these entries are invalid
    flat_tensor = torch.clamp(flat_tensor, 0, num_classes-1) # clamp the input so it can be fed to the one_hot function
    one_hot_tensor = one_hot(flat_tensor, num_classes=num_classes) # actual one_hot
    one_hot_tensor[invalid_values] = torch.zeros(num_classes, dtype=torch.long, device=one_hot_tensor.device) # put zeros in the invalid entries
    one_hot_tensor = one_hot_tensor.view(*x.shape, -1) # reshape to original shape

    return one_hot_tensor


def iterbatches(epochs, dataset, feat, gan, early_stop=False):
    
    global best_unique
    global best_valid
    
    for i in range(epochs):
        for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            adjacency_tensor = make_one_hot(torch.from_numpy(batch[0]), num_classes=gan.edges)
            node_tensor = make_one_hot(torch.from_numpy(batch[1]), num_classes=gan.nodes)

            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]: node_tensor}
            
        if (i+1) % 10 == 0 and early_stop:
            generated_data = gan.predict_gan_generator(6400)

            with rdBase.BlockLogs():
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
                        choices=["qm7", "qm9", "bbbp", "lipo", "ppb"])
    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--atoms", nargs="+", type=int, required=True)
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
    _, datasets, _ = dset(save_dir="data/")
    df = pd.DataFrame(data={'smiles': datasets[0].ids})

    print(f"Featurizing data for {args.dataset}...")
    data = df
    smiles = data['smiles'].values
    max_num = args.vertices
    print(f"Max amount of atoms: {max_num}")
    filtered_smiles = [x for x in smiles if Chem.MolFromSmiles(x).GetNumAtoms() <= max_num]
    feat = dc.deepchem.feat.MolGanFeaturizer(
        max_atom_count=max_num,
        atom_labels=args.atoms
    )
    if args.dataset == "qm9":
        # keep a random 5k subset
        inds = np.random.choice(len(filtered_smiles), 5000, replace=False).tolist()
        filtered_smiles = [filtered_smiles[i] for i in inds]
    features = feat.featurize(filtered_smiles)
    # remove non-GraphMatrix features
    features = [x for x in features if isinstance(x, GraphMatrix)]
    print(f"Found {len(features)} molecules with < {max_num} atoms.")
    dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features],
                                   [x.node_features for x in features])

    out_csv = {"seed": [],
               "validity": [],
               "uniqueness": [],
               "novelty": [],
               "synth": [],
               "drug": []}
    for cur_seed in range(120, 125):
        np.random.seed(cur_seed)
        torch.manual_seed(cur_seed)

        # train model
        gan = MolGAN(edges=4, 
                     vertices=args.vertices, 
                     nodes=len(args.atoms),
                     embedding_dim=64, 
                     mode=mode, 
                     learning_rate=0.0001, 
                     dropout_rate=dropout, 
                     batch_size=32)
        if args.dataset == "qm9":
            # pretrain
            gan.fit_gan(iterbatches(150, dataset, feat, gan), 
                        generator_steps=generator_steps, 
                        checkpoint_interval=2500)
            # train
            gan.fit_gan(iterbatches(150, dataset, feat, gan, True), 
                        generator_steps=generator_steps, 
                        checkpoint_interval=2500)
        else:
            gan.fit_gan(iterbatches(150, dataset, feat, gan), 
                        generator_steps=generator_steps, 
                        checkpoint_interval=2500)

        print(f"Training for seed {cur_seed} complete.")
        print(f"Generating molecules for seed {cur_seed}...")
        generated_data = gan.predict_gan_generator(6400)

        with rdBase.BlockLogs():
            nmols = feat.defeaturize(generated_data)
            print("{} molecules generated".format(len(nmols)))

            nmols = list(filter(lambda x: x is not None, nmols))
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
        # get other scores, Synthesizability, Druglikeliness, Novelty
        eval_dataset = df
        out_csv["novelty"].append(MolecularMetrics.novel_total_score(nmols, eval_dataset))
        out_csv["synth"].append( MolecularMetrics.synthetic_accessibility_score_scores(nmols).mean())
        out_csv["drug"].append(MolecularMetrics.drugcandidate_scores(nmols, eval_dataset).mean())

    df = pd.DataFrame(out_csv, index=None)
    if not os.path.exists("results/"):
        os.makedirs("results/")
    df.to_csv(f"results/dataset_{args.dataset}_elems_{args.atoms}_atoms_{args.vertices}_mode_{args.mode}_dropout_{args.dropout}_gen_steps_{args.generator_steps}_{args.name}_results.csv")
    
    return


if __name__ == '__main__':
    
    args = parse()
    main(args)
