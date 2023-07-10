"""Example pipiline used to prepare training data for predictive model."""
import argparse
from re import split

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm.contrib import tmap
from tqdm.contrib.concurrent import process_map

import preprocessing


def get_train_data(input_file: str):
    """Reads data from .ri file with the following format:
    <SMILES> <type of stationary phase (number)> <retention index>.
    Converts all SMILES to canonical (RDKit) without stereomers
    and then proceeds with all the preprocessing steps for each of the molecules
    to form a training set.

    Parameters
    ----------
    input_file : str
        .ri file with spaces and tabs as delimiters
    """
    output_name = split(r"[\\/]", input_file)[-1][:-3]

    print("Reading input ...")
    df = pd.read_csv(input_file, sep=r"[\s\t]+", engine="python", header=None)
    df.columns = ["Formula", "RI", "ColType"]
    #molecular formulas ARE SORTED 
    #ORDER =/= ORDER IN FILE!
    unq_form, reverse = np.unique(df["Formula"].to_numpy(), return_inverse=True)

    print("Generating molecules ...")
    #molecular objects ARE SORTED!
    #ORDER =/= ORDER IN FILE!
    mols = np.array(list(tmap(Chem.MolFromSmiles, unq_form)))
    val_mask = mols != None
    mols_val = mols[val_mask]
    df = df[val_mask[reverse]].copy()

    print("Saving valid molecules with canonicalized smiles ...")
    _, reverse = np.unique(df["Formula"].values, return_inverse=True)
    n_unq_val_form = np.array(list(tmap(lambda x: Chem.MolToSmiles(x, isomericSmiles=False),
                   mols_val)))
    df["Formula"] = n_unq_val_form[reverse]
    df.to_csv(f"valid_{output_name}.csv", index=False)

    # test True
    # check=lambda x:Chem.MolToSmiles(Chem.MolFromSmiles(x),isomericSmiles=False)
    # print(np.all(np.array(list(map(check,unq_val_form))==n_unq_val_form)))

    print("Re-generating valid molecules from new smiles ...")
    #formulas and molecules ARE NOT SORTED!
    #ORDER == ORDER IN FILE!
    unq_form = pd.unique(df["Formula"])
    np.save(f"unique_{output_name}", unq_form)
    mols = np.array(list(tmap(Chem.MolFromSmiles, unq_form)))

    # test False
    # check=lambda x:Chem.MolToSmiles(Chem.MolFromSmiles(x),isomericSmiles=False)
    # check2=lambda x:Chem.MolToSmiles(x,isomericSmiles=False)
    # print(np.all(np.array(list(map(check2,mols_val))==unq_form)))

    print("Generating molecular descriptors ...")
    md, fp, maccs = [], [], []
    md = process_map(preprocessing.get_descriptors, mols, max_workers=7)
    md_np = np.vstack(md)
    np.save(f"md_{output_name}", md_np)

    print("Generating fingerprints ...")
    fp = process_map(preprocessing.get_fingerprints, mols, max_workers=7)
    fp_np = np.vstack(fp)
    np.save(f"fp_{output_name}", fp_np)

    print("Generating maccs ...")
    maccs = process_map(preprocessing.get_MACCS, mols, max_workers=7)
    maccs_np = np.vstack(maccs)
    np.save(f"maccs_{output_name}", maccs_np)

    #test False
    # print(np.all(np.load("maccs_nist.npy")==np.load("maccs_nist.bak.npy")))

    print("Generating 2d ...")
    twod_arr = process_map(preprocessing.get_2d_coordinates, mols, max_workers=7)
    twod_np = np.stack(twod_arr, axis=0)
    np.save(f"2d_{output_name}", twod_np)
    print("All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare input files for NN.")
    parser.add_argument("input_file", help="Name of input table.")
    args = parser.parse_args()
    get_train_data(args.input_file)
