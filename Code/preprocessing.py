"""Preprocessing module that generates data for all blocks of the model."""

import numpy as np
from mordred import Calculator
from rdkit.Chem import MACCSkeys,MolToMolBlock
from rdkit.Chem.AllChem import GetMorganFingerprint

descriptors = np.load("descriptors.npy", allow_pickle=True)
calc = Calculator(descriptors, ignore_3D=True)


def get_descriptors(mol) -> np.array:
    """Calculates Mordred descriptors for a given RDKit molecule 
    except for the 3D ones and some other that take too much time to calculate.

    Parameters
    ----------
    mol : object
        RDKit molecule

    Returns
    -------
    np.array
        float type array containing all descriptors values. 
        If Mordred has been unable to calculate some descriptors their values are zeroed.
    """
    return np.fromiter(calc(mol).fill_missing(0).values(), dtype=float)


def get_fingerprints(mol, length:int=1024) -> np.array:
    """Calculates Morgan fingerprints (ECFP4) with counts for a given RDKit molecule.

    Parameters
    ----------
    mol : object
        RDKit molecule
    length : int
        Length of int vector

    Returns
    -------
    np.array
        float array containing all fingerprints as their counts
    """
    fingerprints = GetMorganFingerprint(mol, radius=2, useCounts=True)
    byte_vector = np.zeros((length, ))
    for key, val in fingerprints.GetNonzeroElements().items():
        byte_vector[key % length] = val
    return byte_vector


def get_MACCS(mol) -> np.array:
    """Calculates 166 public MACCS keys for a given RDKit molecule.

    Parameters
    ----------
    mol : object
        RDKit molecule

    Returns
    -------
    np.array
        float array with MACCS keys
    """
    return MACCSkeys.GenMACCSKeys(mol)


def get_2d_coordinates(mol) -> np.array:
    """Calculates 2D coordinates for each of the atoms and centers of bonds and encodes them in OneHot manner for a given RDKit molecule.

    Parameters
    ----------
    mol : object
        RDKit molecule

    Returns
    -------
    np.array
        byte 3D array with base as 2D coordinates and additional OneHot  encoded dimension
    """
    SYM = [
        1, 2, 3, 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Si'
    ]
    D_SYM = dict(zip(SYM, range(len(SYM))))
    atoms = []
    bonds = []
    encoded = np.zeros((16, 64, 64), dtype=np.uint8)
    for line in MolToMolBlock(mol).split("\n"):
        if len(line) == 39:
            na, nb = map(int, line.split()[:2])
        elif len(line) == 69:
            atoms.append(line.split()[:4])
        elif len(line) == 12:
            bonds.append(line.split()[:3])
    assert len(atoms) == na, "Wrong number of atoms"
    atoms = np.vstack(atoms)
    atoms_t = atoms[:, -1]
    atoms_xy = atoms[:, :2].astype(float)
    assert len(bonds) == nb, "Wrong number of bonds"
    dx = 32.5 - (max(atoms_xy[:, 0]) + min(atoms_xy[:, 0])) / 2
    dy = 32.5 - (max(atoms_xy[:, 1]) + min(atoms_xy[:, 1])) / 2
    if nb != 0:
        bonds = np.vstack(bonds)
    for i in range(na):
        x, y, = atoms_xy[i, :]
        encoded[D_SYM[atoms_t[i]],
                int(np.round(x + dx, 0)),
                int(np.round(y + dy, 0))] = 1
    for i in range(nb):
        m, n, t = bonds[i, :].astype(int)
        x = (atoms_xy[m - 1, 0] + atoms_xy[n - 1, 0]) / 2
        y = (atoms_xy[m - 1, 1] + atoms_xy[n - 1, 1]) / 2
        encoded[D_SYM[t],
                int(np.round(x + dx, 0)),
                int(np.round(y + dy, 0))] = 1
    return encoded
