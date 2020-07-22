# DonorNumberPrediction

# About
DonorNumberPrediction allows to calculate donor number values using conceptual Density Functional Theory methods.
The basic theory is detailed in: "Calculation of donor numbers: computational estimates for the Lewis basicity of solvents.", R. A. Miranda-Quintana, J. Smiatek; J. Mol. Liquids (submitted)

# License
DonorNumberPrediction is distributed under GPL License version 3 (GPLv3).

# Dependencies
Python >= 3.3;  http://www.python.org/

Numpy >= 1.9.1;  http://www.numpy.org/

SciPy >= 0.11.0;  http://www.scipy.org/

Matplotlib >= 1.0;  http://matplotlib.org/

# Usage
The file donor_number_fit.py contains the class DonorNumber which is used to process the data, fit the models, and estimate their errors.
The file donor_number_out.py contains auxiliary functionality that can be used to process the information contained in a DonorNumber instance.
The experimental DN values must be provided in a separate file containing three columns separated by ";":

inert_solvent;reference_acid;solvent;donor_number

Currently, the ionization energies and electron affinities of the inert solvent, reference acid, and solvents, are included as dictionaries in the donor_number_out.py file.

# Reference
Please, cite both the associated manuscript:

"Calculation of donor numbers: computational estimates for the Lewis basicity of solvents.", R. A. Miranda-Quintana, J. Smiatek; J. Mol. Liquids (submitted)

And this repository:

DOI:
