import numpy as np
from scipy.optimize import curve_fit


class DonorNumber(object):
    """Class to calculate donor numbers.

    Attributes
    ----------
    ref_acid_IA : dict
        Dictionary that contains the ionization energy and electron affinity of the reference
        acid, usually SbCl5.
    solvents_IA : dict
        Dictionary that contains the ionization energies and electron affinities of the solvents.
    inert_solvent_IA : dict
        Dictionary that contains the ionization energies and electron affinities of the
        inert solvent, usually DCE.
    data_file : str
        File containing the experimental data.
    min_bound : float
        Lower bound on the gamma and zeta parameters. Default is no bound.
    max_bound : float
        Upper bound on the gamma and zeta parameters. Default is no bound.
    solvation_data : dict
        Dictionary that contains all the experimental data.
    i_ref_acid : np.ndarray
        Numpy array that contains the available I values for the reference acid that will be used
        in the solvation energy calculations.
    a_ref_acid : np.ndarray
        Numpy array that contains the available A values for the reference acid that will be used
        in the solvation energy calculations.
    i_solv : np.ndarray
        Numpy array that contains the available I values for the solvents that will be used
        in the solvation energy calculations.
    a_solv : np.ndarray
        Numpy array that contains the available A values for the solvents that will be used
        in the solvation energy calculations.
    i_inert_solv : np.ndarray
        Numpy array that contains the available I values for the inert solvent that will be used
        in the solvation energy calculations.
    a_inert_solv : np.ndarray
        Numpy array that contains the available A values for the inert solvent that will be used
        in the solvation energy calculations.
    popt_expl_e_allparams_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the explicit energy calculation.
        All parameters are varied.
    popt_expl_e_allparams_bound : np.ndarray (if it can be calculated)
        Numpy array with the (bound) parameters of the explicit energy calculation.
        All parameters are varied.
    popt_expl_e_gamma_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the explicit energy calculation.
        Only the gamma and linear parameters are varied.
    popt_expl_e_gamma_bound : np.ndarray (if it can be calculated)
        Numpy array with the (bound) parameters of the explicit energy calculation.
        Only the gamma and linear parameters are varied.
    popt_expl_e_zeta_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the explicit energy calculation.
        Only the zeta and linear parameters are varied.
    popt_expl_e_zeta_bound : np.ndarray (if it can be calculated)
        Numpy array with the (bound) parameters of the explicit energy calculation.
        Only the zeta and linear parameters are varied.
    popt_expl_e_simple_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the explicit energy calculation.
        Only the linear parameters are varied.
    popt_impl_e_gamma_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the implicit energy calculation.
        Only the gamma and linear parameters are varied.
    popt_impl_e_gamma_bound : np.ndarray (if it can be calculated)
        Numpy array with the (bound) parameters of the implicit energy calculation.
        Only the gamma and linear parameters are varied.
    popt_impl_e_simple_unbound : np.ndarray (if it can be calculated)
        Numpy array with the (unbound) parameters of the implicit energy calculation.
        Only the linear parameters are varied.
    expl_e_allparams_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the explicit energy calculation.
        Unbound parameters. All parameters varied.
    expl_e_allparams_bound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the explicit energy calculation.
        Bound parameters. All parameters varied.
    expl_e_gamma_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the explicit energy calculation.
        Unbound parameters. Gamma and linear parameters varied.
    expl_e_gamma_bound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the explicit energy calculation.
        Bound parameters. Gamma and linear parameters varied.
    expl_e_zeta_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the explicit energy calculation.
        Unbound parameters. Zeta and linear parameters varied.
    expl_e_zeta_bound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the explicit energy calculation.
        Bound parameters. Zeta and linear parameters varied.
    expl_e_simple_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the explicit energy calculation.
        Unbound parameters. Linear parameters varied.
    impl_e_gamma_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the implicit energy calculations.
        Unbound parameters. Gamma and linear parameters varied.
    impl_e_gamma_bound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the implicit energy calculations.
        Bound parameters. Gamma and linear parameters varied.
    impl_e_simple_unbound_result : np.ndarray (if it can be calculated)
        Numpy array with the results of the implicit energy calculations.
        Unbound parameters. Linear parameters varied.
    expl_e_allparams_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the explicit energy calculations.
        Unbound parameters. All parameters varied.
    expl_e_allparams_bound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the explicit energy calculations.
        Bound parameters. All parameters varied.
    expl_e_gamma_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the explicit energy calculations.
        Unbound parameters. Gamma and linear parameters varied.
    expl_e_gamma_bound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the explicit energy calculations.
        Bound parameters. Gamma and linear parameters varied.
    expl_e_zeta_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the explicit energy calculations.
        Unbound parameters. Zeta and linear parameters varied.
    expl_e_zeta_bound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the explicit energy calculations.
        Bound parameters. Zeta and linear parameters varied.
    expl_e_simple_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the explicit energy calculations.
        Unbound parameters. Linear parameters varied.
    impl_e_gamma_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the implicit energy calculations.
        Unbound parameters. Gamma and linear parameters varied.
    impl_e_gamma_bound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the implicit energy calculations.
        Bound parameters. Gamma and linear parameters varied.
    impl_e_simple_unbound_error : np.ndarray (if it can be calculated)
        Numpy array with the errors of the implicit energy calculations.
        Unbound parameters. Linear parameters varied.

    Methods
    -------
    __init__(self, ref_acid_IA, solvents_IA, inert_solvent_IA, data_file,
             min_bound=-np.inf, max_bound=np.inf)
        Initialize the object.
    gen_data_container()
        Generate the dictionary that will contain all the experimental data.
    read_data(self)
        Read all the experimental data from self.data_file.
    select_data(self)
        Select the I and A values that will be used for the current calculation.
    fit_data(self)
        Fit the data to the experimental DN values.
    get_results_errors(self)
        Calculate the C-DFT DNs and the model errors.
        Errors calculated as: C-DFT_value - Experimental_value.

    Static Methods
    --------------
    Eab(i_a, a_a, i_b, a_b, gamma=1.0, zeta=1.0)
        Calculate the charge transfer energy between two compounds.
    expl_e_allparams(data, gamma_acid, gamma_solv, zeta_acid, zeta_solv, m, b)
        Input of the explicit energy fit, all parameters varied.
    expl_e_gamma(data, gamma_acid, gamma_solv, m, b)
        Input of the explicit energy fit, gamma and linear parameters varied.
    expl_e_zeta(data, zeta_acid, zeta_solv, m, b)
        Input of the explicit energy fit, zeta and linear parameters varied.
    expl_e_simple(data, m, b)
        Input of the explicit energy fit, linear parameters varied.
    impl_e_gamma(data, gamma_acid, gamma_solv, m, b)
        Input of the implicit energy fit, gamma and linear parameters varied.
    impl_e_simple(data, m, b)
        Input of the implicit energy fit, linear parameters varied.
    expl_e_allparams_expl(i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol,
                         gamma_acid, gamma_solv, zeta_acid, zeta_solv, m, b)
        Calculate the DN, all parameters varied.
    expl_e_gamma_expl(i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol,
                      gamma_acid, gamma_solv, m, b)
        Calculate the DN, gamma and linear parameters varied.
    expl_e_zeta_expl(i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol, zeta_acid, zeta_solv, m, b)
        Calculate the DN, zeta and linear parameters varied.
    expl_e_simple_expl(i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol, m, b)
        Calculate the DN, linear parameters varied.
    impl_e_gamma_expl(i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol,
                      gamma_acid, gamma_solv, m, b)
        Calculate the DN, gamma and linear parameters varied.
    impl_e_simple_expl(i_acid, a_acid, i_solv, a_solv, i_in_sol, a_sin_ol, m, b)
        Calculate the DN, linear parameters varied.
    """

    def __init__(self, ref_acid_IA, solvents_IA, inert_solvent_IA, data_file,
                 min_bound=-np.inf, max_bound=np.inf):
        """Initialize the object.

        Parameters
        ----------
        ref_acid_IA : dict
            Dictionary that contains the ionization energy and electron affinity of the reference
            acid, usually SbCl5.
        solvents_IA : dict
            Dictionary that contains the ionization energies and electron affinities of the solvents.
        inert_solvent_IA : dict
            Dictionary that contains the ionization energies and electron affinities of the
            inert solvent, usually DCE.
        data_file : str
            File containing the experimental data.
        min_bound : float
            Lower bound on the gamma and zeta parameters. Default is no bound.
        max_bound : float
            Upper bound on the gamma and zeta parameters. Default is no bound.
        """
        self.ref_acid_IA = ref_acid_IA
        self.solvents_IA = solvents_IA
        self.inert_solvent_IA = inert_solvent_IA
        self.data_file = data_file
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.gen_data_container()
        self.read_data()
        self.select_data()
        self.fit_data()
        self.get_results_errors()

    def gen_data_container(self):
        """Generate the dictionary that will contain all the experimental data."""
        DonorNumber = {}
        for inert_solv in self.inert_solvent_IA:
            DonorNumber[inert_solv] = {}
            for acid in self.ref_acid_IA:
                for solv in self.solvents_IA:
                    DonorNumber[inert_solv][acid + solv] = {}
        self.solvation_data = DonorNumber

    def read_data(self):
        """Read all the experimental data from self.data_file."""
        with open(self.data_file, "r") as infile:
            raw_lines = infile.readlines()
        for line in raw_lines:
            l = line.strip().split(";")
            for inert_solv in self.inert_solvent_IA:
                for acid in self.ref_acid_IA:
                    for solv in self.solvents_IA:
                        if (l[0] == inert_solv) and (l[1] == acid) and (l[2] == solv):
                            if l[-1] != "":
                                self.solvation_data[inert_solv][acid + solv]["DN"] = float(l[-1])
                            else:
                                self.solvation_data[inert_solv][acid + solv]["DN"] = l[-1]

    def select_data(self):
        """Select the I and A values that will be used for the current calculation."""
        i_ref_acid = []
        a_ref_acid = []
        i_solv = []
        a_solv = []
        i_inert_solv = []
        a_inert_solv = []
        exp_data_dn = []
        for inert_solv in self.inert_solvent_IA:
            for acid in self.ref_acid_IA:
                for solv in self.solvents_IA:
                    if isinstance(self.solvation_data[inert_solv][acid + solv]
                                  ["DN"], float):
                        i_ref_acid.append(self.ref_acid_IA[acid][0])
                        a_ref_acid.append(self.ref_acid_IA[acid][1])
                        i_solv.append(self.solvents_IA[solv][0])
                        a_solv.append(self.solvents_IA[solv][1])
                        i_inert_solv.append(self.inert_solvent_IA[inert_solv][0])
                        a_inert_solv.append(self.inert_solvent_IA[inert_solv][1])
                        exp_data_dn.append(self.solvation_data[inert_solv][acid + solv]["DN"])

        self.i_ref_acid = np.array(i_ref_acid)
        self.a_ref_acid = np.array(a_ref_acid)
        self.i_solv = np.array(i_solv)
        self.a_solv = np.array(a_solv)
        self.i_inert_solv = np.array(i_inert_solv)
        self.a_inert_solv = np.array(a_inert_solv)
        self.exp_data_dn = np.array(exp_data_dn)

    @staticmethod
    def Eab(i_a, a_a, i_b, a_b, gamma=1.0, zeta=1.0):
        """Calculate the charge transfer energy between two compounds.

        Parameters
        ----------
        i_a : float
            Ionization energy of compound A.
        a_a : float
            Electron affinity of compound A.
        i_b : float
            Ionization energy of compound B.
        a_b : float
            Electron affinity of compound B.
        gamma : float
            Perturbation on the chemical potential.
        zeta : float
            Perturbation on the hardness.

        Returns
        -------
        dE : float
            Charge transfer energy between two compounds.
        """
        mu_a = -1 * (gamma * i_a + a_a)/(1 + gamma)
        mu_b = -1 * (i_b + gamma * a_b)/(1 + gamma)
        eta_a = zeta * (i_a - a_a)
        eta_b = zeta * (i_b - a_b)
        dE = -0.5 * (mu_a - mu_b)**2 / (eta_a + eta_b)
        return dE

    @staticmethod
    def expl_e_allaparams(data, gamma_acid, gamma_solv, zeta_acid, zeta_solv, m, b):
        """Input of the explicit energy fit, all parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        gamma_acid : float
            Perturbation on the chemical potential of the acid.
        gamma_solv : float
            Perturbation on the chemical potential of the solvent.
        zeta_acid : float
            Perturbation on the hardness of the acid.
        zeta_solv : float
            Perturbation on the hardness of the solvent.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        DN, all parameters varied.
        """
        i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol = data
        dEsolv_cdft = -(DonorNumber.Eab(i_acid, a_acid, i_in_sol, a_in_sol, gamma_acid, zeta_acid) +
                        DonorNumber.Eab(i_solv, a_solv, i_in_sol, a_in_sol, gamma_solv, zeta_solv) -
                        DonorNumber.Eab(i_acid, a_acid, i_solv, a_solv, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def expl_e_gamma(data, gamma_acid, gamma_solv, m, b):
        """Input of the explicit energy fit, gamma and linear parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        gamma_acid : float
            Perturbation on the chemical potential of the acid.
        gamma_solv : float
            Perturbation on the chemical potential of the solvent.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        DN, gamma and linear parameters varied.
        """
        i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol = data
        dEsolv_cdft = -(DonorNumber.Eab(i_acid, a_acid, i_in_sol, a_in_sol, gamma_acid, zeta=1.0) +
                        DonorNumber.Eab(i_solv, a_solv, i_in_sol, a_in_sol, gamma_solv, zeta=1.0) -
                        DonorNumber.Eab(i_acid, a_acid, i_solv, a_solv, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def expl_e_zeta(data, zeta_acid, zeta_solv, m, b):
        """Input of the explicit energy fit, zeta and linear parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        zeta_acid : float
            Perturbation on the hardness of the acid.
        zeta_solv : float
            Perturbation on the hardness of the solvent.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        DN, zeta and linear parameters varied.
        """
        i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol = data
        dEsolv_cdft = -(DonorNumber.Eab(i_acid, a_acid, i_in_sol, a_in_sol, gamma=1.0,
                                        zeta=zeta_acid) +
                        DonorNumber.Eab(i_solv, a_solv, i_in_sol, a_in_sol, gamma=1.0,
                                        zeta=zeta_solv) -
                        DonorNumber.Eab(i_acid, a_acid, i_solv, a_solv, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def expl_e_simple(data, m, b):
        """Input of the explicit energy fit, linear parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        DN, linear parameters varied.
        """
        i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol = data
        dEsolv_cdft = -(DonorNumber.Eab(i_acid, a_acid, i_in_sol, a_in_sol, gamma=1.0, zeta=1.0) +
                        DonorNumber.Eab(i_solv, a_solv, i_in_sol, a_in_sol, gamma=1.0, zeta=1.0) -
                        DonorNumber.Eab(i_acid, a_acid, i_solv, a_solv, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def impl_e_gamma(data, gamma, m, b):
        """Input of the implicit energy fit, gamma and linear parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        gamma_acid : float
            Perturbation on the chemical potential of the acid.
        gamma_solv : float
            Perturbation on the chemical potential of the solvent.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        DN, gamma and linear parameters varied.
        """
        i_acid, a_acid, i_solv, a_solv = data
        dE_impl_e_cdft = DonorNumber.Eab(i_acid, a_acid, i_solv, a_solv, gamma, zeta=1.0)
        return m * dE_impl_e_cdft + b

    @staticmethod
    def impl_e_simple(data, m, b):
        """Input of the implicit energy fit, linear parameters varied.

        Parameters
        ----------
        data : {tuple, list}
            Tuple/list containing the I and A values to be unpacked.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        DN, linear parameters varied.
        """
        i_acid, a_acid, i_solv, a_solv = data
        dE_impl_e_cdft = DonorNumber.Eab(i_acid, a_acid, i_solv, a_solv, gamma=1.0, zeta=1.0)
        return m * dE_impl_e_cdft + b

    @staticmethod
    def expl_e_allaparams_expl(i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol, gamma_acid,
                               gamma_solv, zeta_acid, zeta_solv, m, b):
        """Calculate the DN, all parameters varied.

        Parameters
        ----------
        i_acid : float
            Ionization energy of the acid.
        a_acid : float
            Electron affinity energy of the acid.
        i_solv : float
            Ionization energy of the solvent.
        a_solv : float
            Electron affinity energy of the solvent.
        i_in_sol : float
            Ionization energy of the inert solvent.
        a_in_sol : float
            Electron affinity energy of the inert solvent.
        gamma_acid : float
            Perturbation on the chemical potential of the acid.
        gamma_solv : float
            Perturbation on the chemical potential of the solvent.
        zeta_acid : float
            Perturbation on the hardness of the acid.
        zeta_solv : float
            Perturbation on the hardness of the solvent.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            DN, all parameters varied.
        """
        dEsolv_cdft = -(DonorNumber.Eab(i_acid, a_acid, i_in_sol, a_in_sol, gamma_acid, zeta_acid) +
                        DonorNumber.Eab(i_solv, a_solv, i_in_sol, a_in_sol, gamma_solv, zeta_solv) -
                        DonorNumber.Eab(i_acid, a_acid, i_solv, a_solv, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def expl_e_gamma_expl(i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol, gamma_acid,
                          gamma_solv, m, b):
        """Calculate the DN, gamma and linear parameters varied.

        Parameters
        ----------
        i_acid : float
            Ionization energy of the acid.
        a_acid : float
            Electron affinity energy of the acid.
        i_solv : float
            Ionization energy of the solvent.
        a_solv : float
            Electron affinity energy of the solvent.
        i_in_sol : float
            Ionization energy of the inert solvent.
        a_in_sol : float
            Electron affinity energy of the inert solvent.
        gamma_acid : float
            Perturbation on the chemical potential of the acid.
        gamma_solv : float
            Perturbation on the chemical potential of the solvent.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            DN, gamma and linear parameters varied.
        """
        dEsolv_cdft = -(DonorNumber.Eab(i_acid, a_acid, i_in_sol, a_in_sol, gamma_acid, zeta=1.0) +
                        DonorNumber.Eab(i_solv, a_solv, i_in_sol, a_in_sol, gamma_solv, zeta=1.0) -
                        DonorNumber.Eab(i_acid, a_acid, i_solv, a_solv, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def expl_e_zeta_expl(i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol, zeta_acid,
                         zeta_solv, m, b):
        """Calculate the DN, zeta and linear parameters varied.

        Parameters
        ----------
        i_acid : float
            Ionization energy of the acid.
        a_acid : float
            Electron affinity energy of the acid.
        i_solv : float
            Ionization energy of the solvent.
        a_solv : float
            Electron affinity energy of the solvent.
        i_in_sol : float
            Ionization energy of the inert solvent.
        a_in_sol : float
            Electron affinity energy of the inert solvent.
        zeta_acid : float
            Perturbation on the hardness of the acid.
        zeta_solv : float
            Perturbation on the hardness of the solvent.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            DN, zeta and linear parameters varied.
        """
        dEsolv_cdft = -(DonorNumber.Eab(i_acid, a_acid, i_in_sol, a_in_sol, gamma=1.0,
                                        zeta=zeta_acid) +
                        DonorNumber.Eab(i_solv, a_solv, i_in_sol, a_in_sol, gamma=1.0,
                                        zeta=zeta_solv) -
                        DonorNumber.Eab(i_acid, a_acid, i_solv, a_solv, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def expl_e_simple_expl(i_acid, a_acid, i_solv, a_solv, i_in_sol, a_in_sol, m, b):
        """Calculate the DN, linear parameters varied.

        Parameters
        ----------
        i_acid : float
            Ionization energy of the acid.
        a_acid : float
            Electron affinity energy of the acid.
        i_solv : float
            Ionization energy of the solvent.
        a_solv : float
            Electron affinity energy of the solvent.
        i_in_sol : float
            Ionization energy of the inert solvent.
        a_in_sol : float
            Electron affinity energy of the inert solvent.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            DN, linear parameters varied.
        """
        dEsolv_cdft = -(DonorNumber.Eab(i_acid, a_acid, i_in_sol, a_in_sol, gamma=1.0, zeta=1.0) +
                        DonorNumber.Eab(i_solv, a_solv, i_in_sol, a_in_sol, gamma=1.0, zeta=1.0) -
                        DonorNumber.Eab(i_acid, a_acid, i_solv, a_solv, gamma=1.0, zeta=1.0))
        return m * dEsolv_cdft + b

    @staticmethod
    def impl_e_gamma_expl(i_acid, a_acid, i_solv, a_solv, gamma, m, b):
        """Calculate the DN, gamma and linear parameters varied.

        Parameters
        ----------
        i_acid : float
            Ionization energy of the acid.
        a_acid : float
            Electron affinity energy of the acid.
        i_solv : float
            Ionization energy of the solvent.
        a_solv : float
            Electron affinity energy of the solvent.
        i_in_sol : float
            Ionization energy of the inert solvent.
        a_in_sol : float
            Electron affinity energy of the inert solvent.
        gamma_acid : float
            Perturbation on the chemical potential of the acid.
        gamma_solv : float
            Perturbation on the chemical potential of the solvent.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            DN, gamma and linear parameters varied.
        """
        dE_impl_e_cdft = DonorNumber.Eab(i_acid, a_acid, i_solv, a_solv, gamma, zeta=1.0)
        return m * dE_impl_e_cdft + b

    @staticmethod
    def impl_e_simple_expl(i_acid, a_acid, i_solv, a_solv, m, b):
        """Calculate the DN, linear parameters varied.

        Parameters
        ----------
        i_acid : float
            Ionization energy of the acid.
        a_acid : float
            Electron affinity energy of the acid.
        i_solv : float
            Ionization energy of the solvent.
        a_solv : float
            Electron affinity energy of the solvent.
        i_in_sol : float
            Ionization energy of the inert solvent.
        a_in_sol : float
            Electron affinity energy of the inert solvent.
        m : float
            Slope of the linear fit.
        b : float
            Intercept of the linear fit.

        Returns
        -------
        dEsolv_cdft : float
            DN, linear parameters varied.
        """
        dE_impl_e_cdft = DonorNumber.Eab(i_acid, a_acid, i_solv, a_solv, gamma=1.0, zeta=1.0)
        return m * dE_impl_e_cdft + b

    def fit_data(self):
        """Fit the data to the experimental DN values."""
        data = (self.i_ref_acid, self.a_ref_acid, self.i_solv, self.a_solv,
                self.i_inert_solv, self.a_inert_solv)

        try:
            popt_expl_e_allparams_unbound, _ = curve_fit(DonorNumber.expl_e_allaparams, data,
                                                         self.exp_data_dn,
                                                         bounds=(-np.inf, np.inf), maxfev=10000000)
        except TypeError:
            popt_expl_e_allparams_unbound = None

        try:
            popt_expl_e_allparams_bound, _ = curve_fit(DonorNumber.expl_e_allaparams, data,
                                                       self.exp_data_dn, bounds=((self.min_bound,
                                                                                  self.min_bound,
                                                                                  self.min_bound,
                                                                                  self.min_bound,
                                                                                  -np.inf,
                                                                                  -np.inf),
                                                                                 (self.max_bound,
                                                                                  self.max_bound,
                                                                                  self.max_bound,
                                                                                  self.max_bound,
                                                                                  np.inf,
                                                                                  np.inf)),
                                                       maxfev=10000000)
        except TypeError:
            popt_expl_e_allparams_bound = None

        try:
            popt_expl_e_gamma_unbound, _ = curve_fit(DonorNumber.expl_e_gamma, data,
                                                     self.exp_data_dn, bounds=(-np.inf, np.inf),
                                                     maxfev=10000000)
        except TypeError:
            popt_expl_e_gamma_unbound = None

        try:
            popt_expl_e_gamma_bound, _ = curve_fit(DonorNumber.expl_e_gamma, data,
                                                   self.exp_data_dn, bounds=((self.min_bound,
                                                                              self.min_bound,
                                                                              -np.inf, -np.inf),
                                                                             (self.max_bound,
                                                                              self.max_bound,
                                                                              np.inf, np.inf)),
                                                   maxfev=10000000)
        except TypeError:
            popt_expl_e_gamma_bound = None

        try:
            popt_expl_e_zeta_unbound, _ = curve_fit(DonorNumber.expl_e_zeta, data,
                                                    self.exp_data_dn, bounds=(-np.inf, np.inf),
                                                    maxfev=10000000)
        except TypeError:
            popt_expl_e_zeta_unbound = None

        try:
            popt_expl_e_zeta_bound, _ = curve_fit(DonorNumber.expl_e_zeta, data,
                                                  self.exp_data_dn, bounds=((self.min_bound,
                                                                             self.min_bound,
                                                                             -np.inf, -np.inf),
                                                                            (self.max_bound,
                                                                             self.max_bound,
                                                                             np.inf, np.inf)),
                                                  maxfev=10000000)
        except TypeError:
            popt_expl_e_zeta_bound = None

        try:
            popt_expl_e_simple_unbound, _ = curve_fit(DonorNumber.expl_e_simple, data,
                                                      self.exp_data_dn, bounds=(-np.inf, np.inf),
                                                      maxfev=10000000)
        except TypeError:
            popt_expl_e_simple_unbound = None

        data = (self.i_ref_acid, self.a_ref_acid, self.i_solv, self.a_solv)

        try:
            popt_impl_e_gamma_unbound, _ = curve_fit(DonorNumber.impl_e_gamma, data,
                                                     self.exp_data_dn, bounds=(-np.inf, np.inf),
                                                     maxfev=10000000)
        except TypeError:
            popt_impl_e_gamma_unbound = None

        try:
            popt_impl_e_gamma_bound, _ = curve_fit(DonorNumber.impl_e_gamma, data,
                                                   self.exp_data_dn,
                                                   bounds=((self.min_bound, -np.inf, -np.inf),
                                                           (self.max_bound, np.inf, np.inf)),
                                                   maxfev=10000000)
        except TypeError:
            popt_impl_e_gamma_bound = None

        try:
            popt_impl_e_simple_unbound, _ = curve_fit(DonorNumber.impl_e_simple, data,
                                                      self.exp_data_dn, bounds=(-np.inf, np.inf),
                                                      maxfev=10000000)
        except TypeError:
            popt_impl_e_simple_unbound = None

        if popt_expl_e_allparams_unbound is not None:
            self.popt_expl_e_allparams_unbound = popt_expl_e_allparams_unbound
        if popt_expl_e_allparams_bound is not None:
            self.popt_expl_e_allparams_bound = popt_expl_e_allparams_bound
        if popt_expl_e_gamma_unbound is not None:
            self.popt_expl_e_gamma_unbound = popt_expl_e_gamma_unbound
        if popt_expl_e_gamma_bound is not None:
            self.popt_expl_e_gamma_bound = popt_expl_e_gamma_bound
        if popt_expl_e_zeta_unbound is not None:
            self.popt_expl_e_zeta_unbound = popt_expl_e_zeta_unbound
        if popt_expl_e_zeta_bound is not None:
            self.popt_expl_e_zeta_bound = popt_expl_e_zeta_bound
        if popt_expl_e_simple_unbound is not None:
            self.popt_expl_e_simple_unbound = popt_expl_e_simple_unbound

        if popt_impl_e_gamma_unbound is not None:
            self.popt_impl_e_gamma_unbound = popt_impl_e_gamma_unbound
        if popt_impl_e_gamma_bound is not None:
            self.popt_impl_e_gamma_bound = popt_impl_e_gamma_bound
        if popt_impl_e_simple_unbound is not None:
            self.popt_impl_e_simple_unbound = popt_impl_e_simple_unbound

    def get_results_errors(self):
        """Calculate the C-DFT DN values and errors.

        Notes
        -----
        Errors calculated as: CDFT_value - Experimental_value.
        """
        if hasattr(self, 'popt_expl_e_allparams_unbound'):
            expl_e_allparams_unbound_result = []
            expl_e_allparams_unbound_error = []
            gamma_acid, gamma_solv, zeta_acid, zeta_solv, m, b = self.popt_expl_e_allparams_unbound
            for index, (i_a, a_a, i_s, a_s, i_in_s, a_in_s) in enumerate(
                zip(self.i_ref_acid, self.a_ref_acid, self.i_solv, self.a_solv,
                    self.i_inert_solv, self.a_inert_solv)):
                dE = DonorNumber.expl_e_allaparams_expl(i_a, a_a, i_s, a_s, i_in_s, a_in_s,
                                                        gamma_acid, gamma_solv, zeta_acid,
                                                        zeta_solv, m, b)
                expl_e_allparams_unbound_result.append(dE)
                expl_e_allparams_unbound_error.append(self.exp_data_dn[index] - dE)
            self.expl_e_allparams_unbound_result = np.array(expl_e_allparams_unbound_result)
            self.expl_e_allparams_unbound_error = np.array(expl_e_allparams_unbound_error)

        if hasattr(self, 'popt_expl_e_allparams_bound'):
            expl_e_allparams_bound_result = []
            expl_e_allparams_bound_error = []
            gamma_acid, gamma_solv, zeta_acid, zeta_solv, m, b = self.popt_expl_e_allparams_bound
            for index, (i_a, a_a, i_s, a_s, i_in_s, a_in_s) in enumerate(
                    zip(self.i_ref_acid, self.a_ref_acid, self.i_solv, self.a_solv,
                        self.i_inert_solv, self.a_inert_solv)):
                dE = DonorNumber.expl_e_allaparams_expl(i_a, a_a, i_s, a_s, i_in_s, a_in_s,
                                                        gamma_acid, gamma_solv, zeta_acid,
                                                        zeta_solv, m, b)
                expl_e_allparams_bound_result.append(dE)
                expl_e_allparams_bound_error.append(self.exp_data_dn[index] - dE)
            self.expl_e_allparams_bound_result = np.array(expl_e_allparams_bound_result)
            self.expl_e_allparams_bound_error = np.array(expl_e_allparams_bound_error)

        if hasattr(self, 'popt_expl_e_gamma_unbound'):
            expl_e_gamma_unbound_result = []
            expl_e_gamma_unbound_error = []
            gamma_acid, gamma_solv, m, b = self.popt_expl_e_gamma_unbound
            for index, (i_a, a_a, i_s, a_s, i_in_s, a_in_s) in enumerate(
                    zip(self.i_ref_acid, self.a_ref_acid, self.i_solv, self.a_solv,
                        self.i_inert_solv, self.a_inert_solv)):
                dE = DonorNumber.expl_e_gamma_expl(i_a, a_a, i_s, a_s, i_in_s, a_in_s, gamma_acid,
                                                   gamma_solv, m, b)
                expl_e_gamma_unbound_result.append(dE)
                expl_e_gamma_unbound_error.append(self.exp_data_dn[index] - dE)
            self.expl_e_gamma_unbound_result = np.array(expl_e_gamma_unbound_result)
            self.expl_e_gamma_unbound_error = np.array(expl_e_gamma_unbound_error)

        if hasattr(self, 'popt_expl_e_gamma_bound'):
            expl_e_gamma_bound_result = []
            expl_e_gamma_bound_error = []
            gamma_acid, gamma_solv, m, b = self.popt_expl_e_gamma_bound
            for index, (i_a, a_a, i_s, a_s, i_in_s, a_in_s) in enumerate(
                    zip(self.i_ref_acid, self.a_ref_acid, self.i_solv, self.a_solv,
                        self.i_inert_solv, self.a_inert_solv)):
                dE = DonorNumber.expl_e_gamma_expl(i_a, a_a, i_s, a_s, i_in_s, a_in_s, gamma_acid,
                                                   gamma_solv, m, b)
                expl_e_gamma_bound_result.append(dE)
                expl_e_gamma_bound_error.append(self.exp_data_dn[index] - dE)
            self.expl_e_gamma_bound_result = np.array(expl_e_gamma_bound_result)
            self.expl_e_gamma_bound_error = np.array(expl_e_gamma_bound_error)

        if hasattr(self, 'popt_expl_e_zeta_unbound'):
            expl_e_zeta_unbound_result = []
            expl_e_zeta_unbound_error = []
            zeta_acid, zeta_solv, m, b = self.popt_expl_e_zeta_unbound
            for index, (i_a, a_a, i_s, a_s, i_in_s, a_in_s) in enumerate(
                    zip(self.i_ref_acid, self.a_ref_acid, self.i_solv, self.a_solv,
                        self.i_inert_solv, self.a_inert_solv)):
                dE = DonorNumber.expl_e_zeta_expl(i_a, a_a, i_s, a_s, i_in_s, a_in_s, zeta_acid,
                                                  zeta_solv, m, b)
                expl_e_zeta_unbound_result.append(dE)
                expl_e_zeta_unbound_error.append(self.exp_data_dn[index] - dE)
            self.expl_e_zeta_unbound_result = np.array(expl_e_zeta_unbound_result)
            self.expl_e_zeta_unbound_error = np.array(expl_e_zeta_unbound_error)

        if hasattr(self, 'popt_expl_e_zeta_bound'):
            expl_e_zeta_bound_result = []
            expl_e_zeta_bound_error = []
            zeta_acid, zeta_solv, m, b = self.popt_expl_e_zeta_bound
            for index, (i_a, a_a, i_s, a_s, i_in_s, a_in_s) in enumerate(
                    zip(self.i_ref_acid, self.a_ref_acid, self.i_solv, self.a_solv,
                        self.i_inert_solv, self.a_inert_solv)):
                dE = DonorNumber.expl_e_zeta_expl(i_a, a_a, i_s, a_s, i_in_s, a_in_s, zeta_acid,
                                                  zeta_solv, m, b)
                expl_e_zeta_bound_result.append(dE)
                expl_e_zeta_bound_error.append(self.exp_data_dn[index] - dE)
            self.expl_e_zeta_bound_result = np.array(expl_e_zeta_bound_result)
            self.expl_e_zeta_bound_error = np.array(expl_e_zeta_bound_error)

        if hasattr(self, 'popt_expl_e_simple_unbound'):
            expl_e_simple_unbound_result = []
            expl_e_simple_unbound_error = []
            m, b = self.popt_expl_e_simple_unbound
            for index, (i_a, a_a, i_s, a_s, i_in_s, a_in_s) in enumerate(
                    zip(self.i_ref_acid, self.a_ref_acid, self.i_solv, self.a_solv,
                        self.i_inert_solv, self.a_inert_solv)):
                dE = DonorNumber.expl_e_simple_expl(i_a, a_a, i_s, a_s, i_in_s, a_in_s, m, b)
                expl_e_simple_unbound_result.append(dE)
                expl_e_simple_unbound_error.append(self.exp_data_dn[index] - dE)
            self.expl_e_simple_unbound_result = np.array(expl_e_simple_unbound_result)
            self.expl_e_simple_unbound_error = np.array(expl_e_simple_unbound_error)

        if hasattr(self, 'popt_impl_e_gamma_unbound'):
            impl_e_gamma_unbound_result = []
            impl_e_gamma_unbound_error = []
            gamma, m, b = self.popt_impl_e_gamma_unbound
            for index, (i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_ref_acid, self.a_ref_acid, self.i_solv, self.a_solv)):
                dE = DonorNumber.impl_e_gamma_expl(i_a, a_a, i_s, a_s, gamma, m, b)
                impl_e_gamma_unbound_result.append(dE)
                impl_e_gamma_unbound_error.append(self.exp_data_dn[index] - dE)
            self.impl_e_gamma_unbound_result = np.array(impl_e_gamma_unbound_result)
            self.impl_e_gamma_unbound_error = np.array(impl_e_gamma_unbound_error)

        if hasattr(self, 'popt_impl_e_gamma_bound'):
            impl_e_gamma_bound_result = []
            impl_e_gamma_bound_error = []
            gamma, m, b = self.popt_impl_e_gamma_bound
            for index, (i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_ref_acid, self.a_ref_acid, self.i_solv, self.a_solv)):
                dE = DonorNumber.impl_e_gamma_expl(i_a, a_a, i_s, a_s, gamma, m, b)
                impl_e_gamma_bound_result.append(dE)
                impl_e_gamma_bound_error.append(self.exp_data_dn[index] - dE)
            self.impl_e_gamma_bound_result = np.array(impl_e_gamma_bound_result)
            self.impl_e_gamma_bound_error = np.array(impl_e_gamma_bound_error)

        if hasattr(self, 'popt_impl_e_simple_unbound'):
            impl_e_simple_unbound_result = []
            impl_e_simple_unbound_error = []
            m, b = self.popt_impl_e_simple_unbound
            for index, (i_a, a_a, i_s, a_s) in enumerate(
                    zip(self.i_ref_acid, self.a_ref_acid, self.i_solv, self.a_solv)):
                dE = DonorNumber.impl_e_simple_expl(i_a, a_a, i_s, a_s, m, b)
                impl_e_simple_unbound_result.append(dE)
                impl_e_simple_unbound_error.append(self.exp_data_dn[index] - dE)
            self.impl_e_simple_unbound_result = np.array(impl_e_simple_unbound_result)
            self.impl_e_simple_unbound_error = np.array(impl_e_simple_unbound_error)
