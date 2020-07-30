import numpy as np
from matplotlib import pyplot as plt
from . import trap_analysis
import sympy
from sympy.utilities.lambdify import lambdify, implemented_function
import mpmath
from scipy.interpolate import interp1d
from . import common, kfit

def get_resonator_constants():
    """
    Returns a dictionary of resonator constants used in the calculations in this module.
    :return: {f0, Z0, Q, input power}
    """
    constants = {'f0' : 6.0E9, 'Z0' : 50.0, 'Q' : 10000, 'P' : -100}
    return constants

class ResonatorSolver:

    def __init__(self, use_FEM_data=True):
        self.physical_constants = trap_analysis.get_constants()
        self.resonator_constants = get_resonator_constants()
        self.x = sympy.Symbol('x')
        self.use_FEM_data = True if use_FEM_data else False

        # You either have to specify FEM data points, in the following form:
        # self.x_RF_FEM =
        # self.U_RF_FEM =
        # self.x_DC_FEM =
        # self.V_DC_FEM =
        # Or you can specify the fit values from fits to the potentials as follows:
        # self.dc_params = [a0, a1, a2] from a fit to the DC potential in V
        # self.rf_params = [a0, a1] from a fit to the RF potential in V

    def RF_efield_data(self, xeval):
        Ex_RF = interp1d(self.x_RF_FEM, self.U_RF_FEM, kind='cubic')
        return Ex_RF(xeval)

    def DC_curvature_data(self, xeval):
        DC_curv = interp1d(self.x_DC_FEM, self.V_DC_FEM, kind='cubic')
        return DC_curv(xeval)

    def RF_potential(self, xeval, *p):
        """
        RF potential: U_RF. This is the potential when +/- 0.5 V is applied to the resonator.
        :param xeval: x-point
        :param p: [a0, a1] --> a0 + a1 * x
        :return: a0 + a1 * x
        """
        a0, a1 = p
        f = implemented_function(sympy.Function('f'), lambda y: a0 + a1*y)
        U_RF = lambdify(self.x, f(self.x))
        return U_RF(xeval)

    def RF_efield(self, xeval, *p):
        """
        Derivative of RF_potential
        :param xeval:
        :param p: [a0, a1] --> diff(a0 + a1 * x)
        :return: diff(a0 + a1 * x)
        """
        a0, a1 = p
        return np.float(mpmath.diff(lambda y: a0 + a1*y, xeval))

    def DC_potential(self, xeval, *p):
        """
        :param xeval: x-point
        :param p: [a0, a1, a2] --> a0 + a1*(x-a2)**2
        :return: a0 + a1*(x-a2)**2
        """
        a0, a1, a2 = p
        f = implemented_function(sympy.Function('f'), lambda y: a0 + a1*(y-a2)**2)
        V_DC = lambdify(self.x, f(self.x))
        return V_DC(xeval)

    def DC_curvature(self, xeval, *p):
        """
        :param xeval: x-point
        :param p: [a0, a1, a2] --> diff(diff(a0 + a1 * (x-a2)**2))
        :return: a1
        """
        a0, a1, a2 = p
        return a1

    def setup_eom(self, electron_positions):
        """
        Set up the Matrix used for determining the electron frequency.
        You must make sure to have one of the following:
        if use_FEM_data = True, you must supply RF_efield_data
            - self.x_RF_FEM: x-axis for self.U_RF_FEM (units: m)
            - self.U_RF_FEM: RF E-field (units: V/m)
            - self.x_DC_FEM: x-axis for self.V_DC_FEM (units: m)
            - self.V_DC_FEM: Curvature a1(x), in V_DC(x) = a0 + a1(x-a2)**2 (units: V/m**2)
        if use_FEM_data = False you must supply:
            - self.dc_params: [a0, a1, a2] --> a0 + a1*(x-a2)**2
            - self.rf_params: [a0, a1] --> a0 + a1*x

        :param electron_positions: Electron positions, in the form
                np.array([[x0, x1, ...],
                          [y0, y1, ...)
        :return: M^(-1) * K
        """
        c = self.physical_constants
        r = self.resonator_constants

        omega0 = 2*np.pi*r['f0']
        L = r['Z0']/omega0
        C = 1/(omega0**2 * L)

        num_electrons = np.shape(electron_positions)[1]
        xe, ye = np.array(electron_positions)

        # Set up the inverse of the mass matrix first
        diag_invM = 1/c['m_e'] * np.ones(num_electrons + 1)
        diag_invM[0] = 1/L
        invM = np.diag(diag_invM)

        # Set up the kinetic matrix next
        K = np.zeros(np.shape(invM))
        K[0,0] = 1/C # Bare cavity
        if self.use_FEM_data:
            K[1:,0] = K[0,1:] = c['e']/C * self.RF_efield_data(xe)
        else:
            K[1:,0] = K[0,1:] = c['e']/C * self.RF_efield(xe, self.rf_params) # Coupling terms

        kij_plus = np.zeros((num_electrons, num_electrons))
        for idx in range(num_electrons):
            rij = np.sqrt((xe[idx]-xe)**2 + (ye[idx]-ye)**2)
            tij = np.arctan((ye[idx]-ye)/(xe[idx]-xe))
            kij_plus[idx,:] = 1/4. * c['e']**2/(4*np.pi*c['eps0']) * (1 + 3*np.cos(2*tij))/rij**3

        np.fill_diagonal(kij_plus, 0)

        if self.use_FEM_data:
            K_block = -kij_plus + np.diag(2*c['e']*self.DC_curvature_data(xe) + np.sum(kij_plus, axis=1))
        else:
            K_block = -kij_plus + np.diag(2*c['e']*self.DC_curvature(xe, self.dc_params) + np.sum(kij_plus, axis=1))

        K[1:,1:] = K_block

        return np.dot(invM, K)

    def solve_eom(self, LHS):
        """
        Solves the eigenvalues and eigenvectors for the system of equations constructed with setup_eom()
        :param LHS: matrix product of M^(-1) K
        :return: Eigenvalues, Eigenvectors
        """
        EVals, EVecs = np.linalg.eig(LHS)
        return EVals, EVecs

    def plot_dc_potential(self, x, *p, **kwargs):
        """
        Plot the DC potential with parameters p
        :param x: x-points, unit: micron. May be None if use_FEM_data = True
        :param p: [a0, a1, a2] --> a0 + a1*(x-a2)**2
        :return: None
        """
        if self.use_FEM_data:
            x_interp = np.linspace(np.min(self.x_DC_FEM), np.max(self.x_DC_FEM), 1E4+1)
            plt.plot(self.x_DC_FEM*1E6, self.V_DC_FEM, 'o', **common.plot_opt('darkorange'))
            plt.plot(x_interp*1E6, self.DC_curvature_data(x_interp), '-k', label='Interpolated')
            plt.legend(loc=0)
            plt.title("DC curvature from FEM data")
            plt.xlim(np.min(x_interp*1E6), np.max(x_interp*1E6))
            plt.ylabel(r"$\partial^2 V_\mathrm{dc}/\partial x^2$ ($V$/$m^2$)")
        else:
            plt.plot(x*1E6, self.DC_potential(x, *p), '-', color='darkorange', **kwargs)
            plt.title("DC potential")
            plt.xlim(np.min(x*1E6), np.max(x*1E6))
            plt.ylabel(r"$V_\mathrm{dc}$ (V)")

        plt.xlabel('x ($\mu$m)')

    def plot_rf_potential(self, x, *p, **kwargs):
        """
        Plot the RF potential with parameters p
        :param x: x-points, unit: microns. May be None if use_FEM_data = True
        :param p: [a0, a1] --> a0 + a1*x
        :return: None
        """
        if self.use_FEM_data:
            x_interp = np.linspace(np.min(self.x_RF_FEM), np.max(self.x_RF_FEM), 1E4+1)
            plt.plot(self.x_RF_FEM*1E6, self.U_RF_FEM, 'o', **common.plot_opt('lightblue'))
            plt.plot(x_interp*1E6, self.RF_efield_data(x_interp), '-k', label='Interpolated')
            plt.legend(loc=0)
            plt.title(r"$E_x$ from FEM data")
            plt.xlim(np.min(x_interp)*1E6, np.max(x_interp)*1E6)
            plt.ylabel(r"$E_x$ (V/m)")
        else:
            plt.plot(x*1E6, self.RF_potential(x, *p), '-', color='lightblue', **kwargs)
            plt.title("RF potential")
            plt.xlim(np.min(x)*1E6, np.max(x)*1E6)
            plt.ylabel(r"$U_\mathrm{RF}$ (V/m)")

        plt.xlabel('x ($\mu$m)')