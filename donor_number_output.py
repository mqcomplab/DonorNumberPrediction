import numpy as np
from matplotlib import pyplot as plt
from donor_number_fit import DonorNumber


# Sample file to generate and process the results of a solvation calculation

# Dictionary with the (I, A) pairs for the reference acid, SbCl5 in this case.
ref_acid_IA = {"SbCl5": (8.67, 5.08)}

# Dictionary with the (I, A) pairs for each solvent.
solvents_IA = {"DMSO": (6.44, 0.07), "Tetramethylurea": (6.37, 0.09),
               "Methylpyrrolidinone": (6.64, 0.09), "Methylformamide": (7.17, 0.05),
               "Dimethylformamide": (6.80, 0.15), "Formamide": (7.34, 0.11),
               "Ethylene glycol": (7.56, 0.05), "GBL": (7.54, 0.19),
               "EC": (8.34, 0.27), "DMC": (8.08, -0.05), "DEC": (7.93, -0.04),
               "PC": (8.22, 0.24), "Benzonitrile": (7.53, 1.73),
               "Chloroform": (8.57, 1.41), "Nitromethane": (8.42, 2.26)}

# Dictionary with the (I, A) pairs for the inert solvent, DCE in this case.
inert_solvent_IA = {"DCE": (8.32, 0.58)}


# Auxiliary functions
def gather_data(dn_object):
    """Collect the data from the DonorNumber instance.

    Parameters
    ----------
    dn_object : DonorNumber instance
        DonorNumber object.

    Returns
    -------
    dn_attrs: list of str
        List with the name of the results and errors attributes of the DonorNumber object.
    dn_data : dictionary
        Dictionary with the values of the results and errors attributes of the DonotNumber object.
    """
    dn_data = {}
    dn_attrs = [a for a in dn_object.__dict__.keys() if ("result" in a) or ("error" in a)]
    for attr in dn_attrs:
        dn_data[attr] = eval("dn_object." + attr)
    return dn_attrs, dn_data


def ref_data(dn_object):
    """Collect the reference experimental data from the DonorNumber instance.

    Parameters
    ----------
    dn_object : DonorNumber instance
        DonorNumber object.

    Returns
    -------
    dn_object.exp_data_dn : np.ndarray
        Numpy array with the experimental DN.
    """
    return dn_object.exp_data_dn


def out_str(dn_object, dn_attrs, dn_data, ref_solv):
    """Generate output files."""
    state_function = "DN"

    def aic_variants(attr, errors):
        if "allparams" in attr:
            K = 6
        elif "simple" in attr:
            K = 2
        elif ("expl_e_gamma" in attr) or ("expl_e_zeta" in attr):
            K = 4
        elif "impl_e_gamma" in attr:
            K = 3
        n = len(errors)
        rss = np.sum(errors**2)
        aic = 2 * K + n * np.log(rss/n)
        aic_c = aic + 2 * K * (K + 1)/(n - K - 1)
        return aic, aic_c
        
    s = "RESULTS\n\n"
    s += "For the following inert solvent:\n"
    for in_solvent in dn_object.inert_solvent_IA:
        s += "{}  ".format(in_solvent)
    s += "\n"
    if isinstance(dn_object.ref_acid_IA, dict):
        ref_acids = dn_object.ref_acid_IA.keys()
    else:
        ref_acids = dn_object.ref_acid_IA
    if isinstance(dn_object.solvents_IA, dict):
        solvents = dn_object.solvents_IA.keys()
    else:
        solvents = dn_object.solvents_IA
    s += "We considered the following reference acid and solvents:\n"
    s += "Reference Acid: "
    for acid in ref_acids:
        s += "{}  ".format(acid)
    s += "\nSolvents: "
    for solvent in solvents:
        s += "{}  ".format(solvent)
    s += "\n\n{}_explicit\n              ".format(state_function)
    for attr in sorted(dn_attrs[::2]):
        if "expl_e" in attr:
            parts = attr.split("_")
            header = parts[-3] + "_" + parts[-2]
            s += "{:^32}  ".format(header)
    s += "\nRef {}      ".format(state_function)
    for attr in sorted(dn_attrs):
        if "expl_e" in attr:
            parts = attr.split("_")
            s += "{:^12}     ".format(parts[-1])
    s += "\n"
    for j in range(len(ref_solv)):
        s += "{:>8.3f} ".format(ref_solv[j])
        for attr in sorted(dn_attrs):
            if "expl_e" in attr:
                s += "{:>12.3f}     ".format(dn_data[attr][j])
        s += "\n"
    s += "\nStatistical Summary {}\n              ".format(state_function)
    for attr in sorted(dn_attrs[::2]):
        if "expl_e" in attr:
            parts = attr.split("_")
            header = parts[-3] + "_" + parts[-2]
            s += "{:^32}  ".format(header)
    s += "\n"
    s += "Unsigned Error"
    for attr in sorted(dn_attrs):
        if "expl_e" in attr and "error" in attr:
            s += "{:^32.3f}  ".format(np.sum(dn_data[attr]))
    s += "\n"
    s += "RMSD          "
    for attr in sorted(dn_attrs):
        if "expl_e" in attr and "error" in attr:
            s += "{:^32.3f}  ".format(np.sqrt(np.mean(dn_data[attr]**2)))
    s += "\n"
    aic_values = []
    aic_c_values = []
    for attr in sorted(dn_attrs):
        if "expl_e" in attr and "error" in attr:
            aic, aic_c = aic_variants(attr, dn_data[attr])
            aic_values.append(aic)
            aic_c_values.append(aic_c)
    aic_values = np.array(aic_values)
    aic_c_values = np.array(aic_c_values)
    d_aic_values = aic_values - np.min(aic_values)
    d_aic_c_values = aic_c_values - np.min(aic_c_values)
    indices = []
    for attr in sorted(dn_attrs):
        if "expl_e" in attr and "error" in attr:
            indices.append(1)
    s += "AIC           "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(aic_values[i])
    s += "\n"
    s += "dAIC          "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(d_aic_values[i])
    s += "\n"
    s += "AICc         "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(aic_c_values[i])
    s += "\n"
    s += "dAICc        "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(d_aic_c_values[i])
    s += "\n"
    
    s += "\n\n{}_implicit\n             ".format(state_function)
    for attr in sorted(dn_attrs[::2]):
        if "impl_e" in attr:
            parts = attr.split("_")
            header = parts[-3] + "_" + parts[-2]
            s += "{:^32}  ".format(header)
    s += "\nRef {}       ".format(state_function)
    for attr in sorted(dn_attrs):
        if "impl_e" in attr:
            parts = attr.split("_")
            s += "{:^12}     ".format(parts[-1])
    s += "\n"
    for j in range(len(ref_solv)):
        s += "{:>8.3f} ".format(ref_solv[j])
        for attr in sorted(dn_attrs):
            if "impl_e" in attr:
                s += "{:>12.3f}     ".format(dn_data[attr][j])
        s += "\n"
    s += "\nStatistical Summary  {}\n             ".format(state_function)
    for attr in sorted(dn_attrs[::2]):
        if "impl_e" in attr:
            parts = attr.split("_")
            header = parts[-3] + "_" + parts[-2]
            s += "{:^32}  ".format(header)
    s += "\n"
    s += "Unsigned Error"
    for attr in sorted(dn_attrs):
        if "impl_e" in attr and "error" in attr:
            s += "{:^32.3f}  ".format(np.sum(dn_data[attr]))
    s += "\n"
    s += "RMSD          "
    for attr in sorted(dn_attrs):
        if "impl_e" in attr and "error" in attr:
            s += "{:^32.3f}  ".format(np.sqrt(np.mean(dn_data[attr]**2)))
    s += "\n"
    aic_values = []
    aic_c_values = []
    for attr in sorted(dn_attrs):
        if "impl_e" in attr and "error" in attr:
            aic, aic_c = aic_variants(attr, dn_data[attr])
            aic_values.append(aic)
            aic_c_values.append(aic_c)
    aic_values = np.array(aic_values)
    aic_c_values = np.array(aic_c_values)
    d_aic_values = aic_values - np.min(aic_values)
    d_aic_c_values = aic_c_values - np.min(aic_c_values)
    indices = []
    for attr in sorted(dn_attrs):
        if "impl_e" in attr and "error" in attr:
            indices.append(1)
    s += "AIC           "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(aic_values[i])
    s += "\n"
    s += "dAIC          "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(d_aic_values[i])
    s += "\n"
    s += "AICc         "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(aic_c_values[i])
    s += "\n"
    s += "dAICc        "
    for i in range(len(indices)):
        s += "{:^32.3f}  ".format(d_aic_c_values[i])
    s += "\n"
    
    with open("DN_Results.txt", "w") as outfile:
        outfile.write(s)


def gen_individual_pic(state_function, name_data, x_label, y_label, attr, ref_values, cdft_values):
    """Generate the figure for a single result."""
    fig, ax = plt.subplots()
    ax.scatter(cdft_values, ref_values, s=25, cmap=plt.cm.ocean, zorder=10)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.title.set_text(name_data)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.savefig("{}_{}.png".format(state_function, attr[:-7]))
    plt.cla()


def gen_pics(dn_object, dn_attrs, dn_data, ref_solv):
    """Generate the figures for all the results."""
    state_function = "DN"
    for attr in sorted(dn_attrs):
        if "error" in attr:
            pass
        else:
            plt.close()
            if "expl_e" in attr:
                ref_values = ref_solv
                x_label = "DN_expl_CDFT"
                y_label = "{}_experimental".format(state_function)
            else:
                ref_values = ref_solv
                x_label = "DN_impl_CDFT"
                y_label = "{}_experimental".format(state_function)
            cdft_values = dn_data[attr]
            
            name_data = "{}_{}\n".format(state_function, attr[:-7])
            gen_individual_pic(state_function, name_data, x_label, y_label, attr,
                               ref_values, cdft_values)
            

if __name__ == "__main__":
    # DonorNumber object that will be used to generate the results.
    dn_object = DonorNumber(ref_acid_IA=ref_acid_IA, solvents_IA=solvents_IA,
                            inert_solvent_IA=inert_solvent_IA, data_file="DN_data.csv", min_bound=0)

    dn_attrs, dn_data = gather_data(dn_object)
    ref_dn = ref_data(dn_object)
    out_str(dn_object, dn_attrs, dn_data, ref_dn)
    gen_pics(dn_object, dn_attrs, dn_data, ref_dn)
