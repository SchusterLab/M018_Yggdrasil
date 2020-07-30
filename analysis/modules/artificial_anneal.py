import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import approx_fprime, minimize
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
import os, time, functools, multiprocessing, dxfgrabber
from termcolor import cprint
from .import_data import load_dsp
from . import common
from . import interpolate_slow

class ConvergenceMonitor:
    def __init__(self, Uopt, grad_Uopt, N, Uext=None, xext=None, yext=None, verbose=True, eps=1E-12, save_path=None,
                 figsize=(6.5,3.), coordinate_transformation=None, clim=(-0.75, 0)):
        """
        To be used with scipy.optimize.minimize as a call back function. One has two choices for call-back functions:
        - monitor_convergence: print the status of convergence (value of Uopt and norm of grad_Uopt)
        - save_pictures: save figures of the electron positions every iteration to construct a movie.
        :param Uopt: Cost function or total energy of the system. Uopt takes one argument and returns a scalar.
        :param grad_Uopt: Gradient of Uopt. This should be a function that takes one argument and returns
                          an array of size 2*N_electrons
        :param N: Report the status of optimization every N times. This should be an integer
        :param Uext: Electrostatic potential function. Takes 2 arguments (x,y) and returns an array of the size of x
        and y
        :param xext: Array of arbitrary size for evaluating the electrostatic potential. Units should be meters.
        :param yext: Array of arbitrary size for evaluating the electrostatic potential. Units should be meters.
        :param verbose: Whether to print the status when monitor_convergence is called.
        :param eps: Step size used to numerically approximate the gradient with scipy.optimize.approx_fprime
        :param save_path: Directory in which to save figures when self.save_pictures is called. None by default.
        """
        self.clim = clim
        self.call_every = N
        self.call_counter = 0
        self.verbose = verbose
        self.curr_grad_norm = list()
        self.curr_fun = list()
        self.iter = list()
        self.epsilon = eps
        self.save_path = save_path
        self.Uopt = Uopt
        self.grad_Uopt = grad_Uopt
        self.xext, self.yext, self.Uext = xext, yext, Uext
        self.figsize = figsize
        self.coordinate_transformation = coordinate_transformation
        self.electrode_outline_filename = None

    def monitor_convergence(self, xk):
        """
        Monitor the convergence while the optimization is running. To be used with scipy.optimize.minimize.
        :param xk: Electron position pairs
        :return: None
        """
        if not (self.call_counter % self.call_every):
            self.iter.append(self.call_counter)
            self.curr_fun.append(self.Uopt(xk))

            # Here we use the L-inf norm (the maximum)
            self.curr_grad_norm.append(np.max(np.abs(self.grad_Uopt(xk))))

            if self.call_counter == 0:
                self.curr_xk = xk
                self.jac = self.grad_Uopt(xk)
                #self.approx_fprime = approx_fprime(xk, self.Uopt, self.epsilon)
            else:
                self.curr_xk = np.vstack((self.curr_xk, xk))
                self.jac = np.vstack((self.jac, self.grad_Uopt(xk)))
                #self.approx_fprime = np.vstack((self.approx_fprime, approx_fprime(xk, self.Uopt, self.epsilon)))

            if self.verbose:
                print("%d\tUopt: %.8f eV\tNorm of gradient: %.2e eV/m" \
                      % (self.call_counter, self.curr_fun[-1], self.curr_grad_norm[-1]))

        self.call_counter += 1

    def save_pictures(self, xk):
        """
        Plots the current value of the electron position array xk and saves a picture in self.save_path.
        :param xk: Electron position pairs
        :return: None
        """
        xext, yext = self.xext, self.yext
        Uext = self.Uext

        fig = plt.figure(figsize=self.figsize)
        try:
            common.configure_axes(12)
        except:
            pass

        if (Uext is not None) and (xext is not None) and (yext is not None):
            Xext, Yext = np.meshgrid(xext, yext)
            plt.pcolormesh(xext * 1E6, yext * 1E6, Uext(Xext, Yext), cmap=plt.cm.RdYlBu, vmax=self.clim[1], vmin=self.clim[0])
            plt.xlim(np.min(xext) * 1E6, np.max(xext) * 1E6)
            plt.ylim(np.min(yext) * 1E6, np.max(yext) * 1E6)

        if self.coordinate_transformation is None:
            electrons_x, electrons_y = xk[::2], xk[1::2]
        else:
            r_new = self.coordinate_transformation(xk)
            electrons_x, electrons_y = r2xy(r_new)

        plt.plot(electrons_x*1E6, electrons_y*1E6, 'o', color='deepskyblue')
        plt.xlabel("$x$ ($\mu$m)")
        plt.ylabel("$y$ ($\mu$m)")
        plt.colorbar()

        if self.electrode_outline_filename is not None:
            PP = PostProcess(save_path=self.save_path)
            PP.draw_from_dxf(filename=self.electrode_outline_filename, color='black', alpha=0.6, lw=0.5)

        if self.save_path is not None:
            common.save_figure(fig, save_path=self.save_path)
            #fig.savefig(os.path.join(self.save_path, '%.5d.png' % (self.call_counter)), bbox_inches='tight', dpi=300)
        else:
            print("Please specify a save path when initiating ConvergenceMonitor")

        plt.close(fig)

        self.monitor_convergence(xk)

    def create_movie(self, fps, filenames_in="%05d.png", filename_out="movie.mp4"):
        """
        Generate a movie from the pictures generated by save_pictures. Movie gets saved in self.save_path
        For filenames of the type 00000.png etc use filenames_in="%05d.png".
        Files must all have the save extension and resolution.
        :param fps: frames per second (integer).
        :param filenames_in: Signature of series of file names in Unix style. Ex: "%05d.png"
        :param filename_out: File name of the output video. Ex: "movie.mp4"
        :return: None
        """
        curr_dir = os.getcwd()
        os.chdir(self.save_path)
        os.system(r"ffmpeg -r {} -b 1800 -i {} {}".format(int(fps), filenames_in, filename_out))
        os.chdir(curr_dir)

class PostProcess:
    def __init__(self, save_path=None, res_pinw=0.9, edge_gapw=0.5, center_gapw=0.7, box_length=40):
        """
        Post process the results after the optimizer has converged.
        :param save_path: Directory in which to save figures/movies.
        """
        self.save_path = save_path
        self.trapped_electrons = None
        self.res_pinw = res_pinw
        self.edge_gapw = edge_gapw
        self.center_gapw = center_gapw
        self.box_length = box_length

    def draw_from_dxf(self, filename, offset, **plot_options):
        draw_from_dxf(filename, offset=offset, **plot_options)

    def draw_resonator_pins(self):
        box_y_length = self.box_length
        pin_width = self.res_pinw
        edge_gap = self.edge_gapw
        center_gap = self.center_gapw

        pin1_x = [-(center_gap / 2. + pin_width), -center_gap / 2., -center_gap / 2., -(center_gap / 2. + pin_width),
                  -(center_gap / 2. + pin_width)]
        pin1_y = [-box_y_length / 2., -box_y_length / 2., box_y_length / 2., box_y_length / 2., -box_y_length / 2.]

        pin2_x = [(center_gap / 2. + pin_width), center_gap / 2., center_gap / 2., (center_gap / 2. + pin_width),
                  (center_gap / 2. + pin_width)]
        pin2_y = [-box_y_length / 2., -box_y_length / 2., box_y_length / 2., box_y_length / 2., -box_y_length / 2.]

        plt.plot(pin1_x, pin1_y, lw=0)
        #plt.plot(pin2_x, pin2_y, **plot_kwargs)
        #plt.plot([-(edge_gap + pin_width + center_gap / 2.), -(edge_gap + pin_width + center_gap / 2.)],
        #         [-box_y_length / 2., box_y_length / 2.], **plot_kwargs)
        #plt.plot([(edge_gap + pin_width + center_gap / 2.), (edge_gap + pin_width + center_gap / 2.)],
        #         [-box_y_length / 2., box_y_length / 2.], **plot_kwargs)

        plt.fill_between(pin1_x, pin1_y, y2=-box_y_length / 2.,
                         color='none', hatch='X', edgecolor='k', alpha=0.5)
        plt.fill_between(pin2_x, pin2_y, y2=-box_y_length / 2.,
                         color='none', hatch='X', edgecolor='k', alpha=0.5)
        plt.fill_between([-(edge_gap + pin_width + center_gap / 2. + 3.0), -(edge_gap + pin_width + center_gap / 2.)],
                        [box_y_length / 2., box_y_length / 2.], y2=[-box_y_length / 2., -box_y_length/2.],
                        color='none', hatch='X', edgecolor='k', alpha=0.5)
        plt.fill_between([(edge_gap + pin_width + center_gap / 2.), (edge_gap + pin_width + center_gap / 2. + 3.0)],
                        [box_y_length / 2., box_y_length / 2.], y2=[-box_y_length / 2., -box_y_length / 2.],
                        color='none', hatch='X', edgecolor='k', alpha=0.5)

    def get_electron_density(self, r, verbose=True):
        """
        Calculate the electron density based on the nearest neighbor for each electron. The electron density is in m^-2
        :param r: Electron x,y coordinate pairs
        :param verbose: Whether to print the result or just to return it
        :return: Electron density in m^-2
        """
        xi, yi = r2xy(r)
        Xi, Yi = np.meshgrid(xi, yi)
        Xj, Yj = Xi.T, Yi.T

        Rij = np.sqrt((Xi - Xj) ** 2 + (Yi - Yj) ** 2)
        np.fill_diagonal(Rij, 1E10)

        nearest_neighbors = np.min(Rij, axis=1)
        ns = 1 / (np.mean(nearest_neighbors)) ** 2
        if verbose:
            print("The electron density in the figure above is %.3e m^-2" % ns)

        return ns

    def save_snapshot(self, r, xext=None, yext=None, Uext=None, figsize=(12.,3.), clim=(-1,0), common=None, title="",
                      draw_resonator_pins=True, draw_from_dxf={'filename' : None, 'plot_options' : {'color' : 'black'}}):
        """
        Save a picture with the electron positions on top of the electrostatic potential data.
        :param r: Electron x,y coordinate pairs
        :param xext: x-data (1D-array) for plotting the electrostatic potential
        :param yext: y-data (1D-array) for plotting the electrostatic potential
        :param Uext: Potential function for plotting the electrostatic potential
        :param figsize: Tuple that regulates the figure size
        :param common: module that allows saving
        :return: None
        """
        fig = plt.figure(figsize=figsize)

        if common is not None:
            common.configure_axes(12)

        if (Uext is not None) and (xext is not None) and (yext is not None):
            Xext, Yext = np.meshgrid(xext, yext)
            plt.pcolormesh(xext * 1E6, yext * 1E6, Uext(Xext, Yext), cmap=plt.cm.Spectral_r,
                           vmax=clim[1], vmin=clim[0])
            plt.xlim(np.min(xext) * 1E6, np.max(xext) * 1E6)
            plt.ylim(np.min(yext) * 1E6, np.max(yext) * 1E6)

        if draw_resonator_pins:
            self.draw_resonator_pins()
        if draw_from_dxf['filename'] is not None:
            self.draw_from_dxf(draw_from_dxf['filename'], draw_from_dxf['offset'],
                               **draw_from_dxf['plot_options'])

        plt.plot(r[::2] * 1E6, r[1::2] * 1E6, 'o', color='deepskyblue')
        plt.xlabel("$x$ ($\mu$m)")
        plt.ylabel("$y$ ($\mu$m)")

        plt.colorbar()
        plt.title(title)

        if self.save_path is not None and common is not None:
            common.save_figure(fig, save_path=self.save_path)
        else:
            print("Please specify a save path when initiating PostProcess")

        plt.close('all')

    def write2file(self, **kwargs):
        """
        Write simulation results to an npz file in self.save_path
        :param kwargs: Dictionary of parameters to be saved to the file.
        :return: None
        """

        number = 0
        file_name = "%.5d.npz"%number

        while file_name in os.listdir(self.save_path):
            file_name = "%.5d.npz"%(number)
            number += 1

        #print("Saving file to %s ..."%(os.path.join(self.save_path, file_name)))

        np.savez(os.path.join(self.save_path, file_name), **kwargs)

    def get_trapped_electrons(self, r, trap_area_x=(-4E-6, -1.8E-6)):
        """
        Evaluate how many electrons are in the area specified by the bounds trap_area_x[0] < x < trap_area_x[1]
        :param r: r = np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
        :param trap_area_x: Tuple specifying the bounds of the trapping area
        :return: The number of electrons in the trapping area (scalar)
        """
        return len(np.where(np.logical_and(r[::2] > trap_area_x[0], r[::2] < trap_area_x[1]))[0])

class TrapAreaSolver:

    def __init__(self, grid_data_x, grid_data_y, potential_data, spline_order_x=3, spline_order_y=3, smoothing=0,
                 include_screening=True, screening_length=2*0.8E-6):
        """
        This class is used for constructing the functional forms required for scipy.optimize.minimize.
        It deals with the Maxwell input data, as well as constructs the cost function used in the optimizer.
        It also calculates the gradient, that can be used to speed up the optimizer.
        :param grid_data_x: 1D array of x-data. Coordinates from grid_data_x & grid_data_y must form a rectangular grid
        :param grid_data_y: 1D array of y-data. Coordinates from grid_data_x & grid_data_y must form a rectangular grid
        :param potential_data: Energy land scape, - e V_ext.
        :param spline_order_x: Order of the interpolation in the x-direction (1 = linear, 3 = cubic)
        :param spline_order_y: Order of the interpolation in the y-direction (1 = linear, 3 = cubic)
        :param smoothing: Absolute smoothing. Effect depends on scale of potential_data.
        """
        self.interpolator = RectBivariateSpline(grid_data_x, grid_data_y, potential_data,
                                                kx=spline_order_x, ky=spline_order_y, s=smoothing)

        # Constants
        self.include_screening = include_screening
        self.screening_length = screening_length
        self.qe = 1.602E-19
        self.eps0 = 8.85E-12
        self.kB = 1.38E-23

    def V(self, xi, yi):
        """
        Evaluate the electrostatic potential at coordinates xi, yi
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return: Interpolated value(s) of the data supplied to __init__ at values (xi, yi)
        """
        return self.interpolator.ev(xi, yi)

    def Velectrostatic(self, xi, yi):
        """
        When supplying two arrays of size n to V, it returns an array
        of size nxn, according to the meshgrid it has evaluated. We're only interested
        in the sum of the diagonal elements, so we take the sum and this represents
        the sum of the static energy of the n particles in the potential.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        """
        return self.qe * np.sum(self.V(xi, yi))

    def Vee(self, xi, yi, eps=1E-15):
        """
        Returns the repulsion potential between two electrons separated by a distance sqrt(|xi-xj|**2 + |yi-yj|**2)
        Note the factor 1/2. in front of the potential energy to avoid overcounting.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        """
        Xi, Yi = np.meshgrid(xi, yi)
        Xj, Yj = Xi.T, Yi.T

        Rij = np.sqrt((Xi - Xj) ** 2 + (Yi - Yj) ** 2)
        np.fill_diagonal(Rij, eps)

        if self.include_screening:
            return + 1 / 2. * self.qe ** 2 / (4 * np.pi * self.eps0) * np.exp(-Rij/self.screening_length) / Rij
        else:
            return + 1 / 2. * self.qe ** 2 / (4 * np.pi * self.eps0) * 1 / Rij

    def Vtotal(self, r):
        """
        This can be used as a cost function for the optimizer.
        Returns the total energy of N electrons
        r is a 0D array with coordinates of the electrons.
        The x-coordinates are thus given by the even elements of r: r[::2],
        whereas the y-coordinates are the odd ones: r[1::2]
        :param r: r = np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
        :return: Scalar with the total energy of the system.
        """
        xi, yi = r[::2], r[1::2]
        Vtot = self.Velectrostatic(xi, yi)
        interaction_matrix = self.Vee(xi, yi)
        np.fill_diagonal(interaction_matrix, 0)
        Vtot += np.sum(interaction_matrix)
        return Vtot / self.qe

    def dVdx(self, xi, yi):
        """
        Derivative of the electrostatic potential in the x-direction.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=1, dy=0)

    def ddVdx(self, xi, yi):
        """
        Second derivative of the electrostatic potential in the x-direction.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=2, dy=0)

    def dVdy(self, xi, yi):
        """
        Derivative of the electrostatic potential in the y-direction
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=0, dy=1)

    def ddVdy(self, xi, yi):
        """
        Second derivative of the electrostatic potential in the y-direction.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=0, dy=2)

    def ddVdxdy(self, xi, yi):
        """
        Second derivative of the electrostatic potential in the y-direction.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=1, dy=1)

    def grad_Vee(self, xi, yi, eps=1E-15):
        """
        Derivative of the electron-electron interaction term
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :param eps: A small but non-zero number to avoid triggering Warning message. Exact value is irrelevant.
        :return: 1D-array of size(xi) + size(yi)
        """
        Xi, Yi = np.meshgrid(xi, yi)
        Xj, Yj = Xi.T, Yi.T

        Rij = np.sqrt((Xi - Xj) ** 2 + (Yi - Yj) ** 2)
        np.fill_diagonal(Rij, eps)

        gradx_matrix = np.zeros(np.shape(Rij))
        grady_matrix = np.zeros(np.shape(Rij))
        gradient = np.zeros(2 * len(xi))

        if self.include_screening:
            gradx_matrix = -1 * self.qe ** 2 / (4 * np.pi * self.eps0) * np.exp(-Rij/self.screening_length) * \
                           (Xi - Xj) * (Rij + self.screening_length) / (self.screening_length * Rij ** 3)
            grady_matrix = +1 * self.qe ** 2 / (4 * np.pi * self.eps0) * np.exp(-Rij/self.screening_length) * \
                           (Yi - Yj) * (Rij + self.screening_length) / (self.screening_length * Rij ** 3)
        else:
            gradx_matrix = -1 * self.qe ** 2 / (4 * np.pi * self.eps0) * (Xi - Xj) / Rij ** 3
            grady_matrix = +1 * self.qe ** 2 / (4 * np.pi * self.eps0) * (Yi - Yj) / Rij ** 3


        np.fill_diagonal(gradx_matrix, 0)
        np.fill_diagonal(grady_matrix, 0)

        gradient[::2] = np.sum(gradx_matrix, axis=0)
        gradient[1::2] = np.sum(grady_matrix, axis=0)

        return gradient

    def grad_total(self, r):
        """
        Total derivative of the cost function. This may be used in the optimizer to converge faster.
        :param r: r = np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
        :return: 1D array of length len(r), where grad_total = np.array([dV/dx|r0, dV/dy|r0, ...])
        """
        xi, yi = r[::2], r[1::2]
        gradient = np.zeros(len(r))
        gradient[::2] = self.dVdx(xi, yi)
        gradient[1::2] = self.dVdy(xi, yi)
        gradient += self.grad_Vee(xi, yi) / self.qe
        return gradient

    def thermal_kick_x(self, x, y, T, maximum_dx=None):
        ktrapx = np.abs(self.qe * self.ddVdx(x, y))
        ret = np.sqrt(2 * self.kB * T / ktrapx)
        if maximum_dx is not None:
            ret[ret > maximum_dx] = maximum_dx
            return ret
        else:
            return ret

    def thermal_kick_y(self, x, y, T, maximum_dy=None):
        ktrapy = np.abs(self.qe * self.ddVdy(x, y))
        ret = np.sqrt(2 * self.kB * T / ktrapy)
        if maximum_dy is not None:
            ret[ret > maximum_dy] = maximum_dy
            return ret
        else:
            return ret

    def single_thread(self, iteration, electron_initial_positions, T, cost_function, minimizer_dict, maximum_dx, maximum_dy):
        xi, yi = r2xy(electron_initial_positions)
        np.random.seed(np.int(time.time()) + iteration)
        xi_prime = xi + self.thermal_kick_x(xi, yi, T, maximum_dx=maximum_dx) * np.random.randn(len(xi))
        yi_prime = yi + self.thermal_kick_y(xi, yi, T, maximum_dy=maximum_dy) * np.random.randn(len(yi))
        electron_perturbed_positions = xy2r(xi_prime, yi_prime)
        return minimize(cost_function, electron_perturbed_positions, **minimizer_dict)

    def parallel_perturb_and_solve(self, cost_function, N_perturbations, T, solution_data_reference, minimizer_dict,
                                   maximum_dx=None, maximum_dy=None):
        """
        This function is to be run after a minimization by scipy.optimize.minimize has already occured.
        It takes the output of that function in solution_data_reference and tries to find a lower energy state
        by perturbing the system N_perturbation times at temperature T. See thermal_kick_x and thermal_kick_y.
        This function runs N_perturbations in parallel on the cores of your CPU.
        :param cost_function: A function that takes the electron positions and returns the total energy
        :param N_perturbations: Integer, number of perturbations to find a new minimum
        :param T: Temperature to perturb the system at. This is used to convert to a motion.
        :param solution_data_reference: output of scipy.optimize.minimize
        :param minimizer_dict: Dictionary with optimizer options. See scipy.optimize.minimize
        :return: output of minimize with the lowest evaluated cost function
        """
        electron_initial_positions = solution_data_reference['x']
        best_result = solution_data_reference
        pool = multiprocessing.Pool()

        tasks = []
        iteration = 0
        while iteration < N_perturbations:
            iteration += 1
            tasks.append((iteration, electron_initial_positions, T, cost_function, minimizer_dict, maximum_dx, maximum_dy,))

        results = [pool.apply_async(self.single_thread, t) for t in tasks]
        for result in results:
            res = result.get()

            if res['status'] == 0 and res['fun'] < best_result['fun']:
                #cprint("\tNew minimum was found after perturbing!", "green")
                best_result = res

        # Nothing has changed by perturbing the reference solution
        if (best_result['x'] == solution_data_reference['x']).all():
            cprint("Solution data unchanged after perturbing", "white")
        # Or there is a new minimum
        else:
            cprint("Better solution found (%.3f%% difference)" \
                   % (100 * (best_result['fun'] - solution_data_reference['fun']) / solution_data_reference['fun']),
                   "green")


        return best_result

    def perturb_and_solve(self, cost_function, N_perturbations, T, solution_data_reference,
                          maximum_dx=None, maximum_dy=None, do_print=True, **minimizer_options):
        """
        This function is to be run after a minimization by scipy.optimize.minimize has already occured.
        It takes the output of that function in solution_data_reference and tries to find a lower energy state
        by perturbing the system N_perturbation times at temperature T. See thermal_kick_x and thermal_kick_y.
        :param cost_function: A function that takes the electron positions and returns the total energy
        :param N_perturbations: Integer, number of perturbations to find a new minimum
        :param T: Temperature to perturb the system at. This is used to convert to a motion.
        :param solution_data_reference: output of scipy.optimize.minimize
        :param minimizer_options: Dictionary with optimizer options. See scipy.optimize.minimize
        :return: output of minimize with the lowest evaluated cost function
        """
        electron_initial_positions = solution_data_reference['x']
        best_result = solution_data_reference

        for n in range(N_perturbations):
            xi, yi = r2xy(electron_initial_positions)
            xi_prime = xi + self.thermal_kick_x(xi, yi, T, maximum_dx=maximum_dx) * np.random.randn(len(xi))
            yi_prime = yi + self.thermal_kick_y(xi, yi, T, maximum_dy=maximum_dy) * np.random.randn(len(yi))
            electron_perturbed_positions = xy2r(xi_prime, yi_prime)

            res = minimize(cost_function, electron_perturbed_positions, **minimizer_options)

            if res['status'] == 0 and res['fun'] < best_result['fun']:
                if do_print:
                    cprint("\tNew minimum was found after perturbing!", "green")
                best_result = res
            elif res['status'] == 0 and res['fun'] > best_result['fun']:
                pass  # No new minimum was found after perturbation, this is quite common.
            elif res['status'] != 0 and res['fun'] < best_result['fun']:
                if do_print:
                    cprint("\tThere is a lower state, but minimizer didn't converge!", "red")
            elif res['status'] != 0 and res['fun'] > best_result['fun']:
                pass

        return best_result

    def calculate_mu(self, ri):
        electrons_x, electrons_y = r2xy(ri)
        interactions = self.Vee(electrons_x, electrons_y) / self.qe
        np.fill_diagonal(interactions, 0)
        mu = list()
        el = 0
        for electron_x, electron_y in zip(electrons_x, electrons_y):
            mu.append(np.sum(interactions[el, :]) + self.Velectrostatic(electron_x, electron_y) / 1.602E-19)
            el += 1

        return np.array(mu)

class ResonatorSolver:

    def __init__(self, grid_data, potential_data, efield_data=None, box_length=40E-6, spline_order_x=3, smoothing=0,
                 include_screening=True, screening_length=2 * 0.8E-6):
        self.interpolator = UnivariateSpline(grid_data, potential_data, k=spline_order_x, s=smoothing, ext=3)
        self.derivative = self.interpolator.derivative(n=1)
        self.second_derivative = self.interpolator.derivative(n=2)
        self.box_y_length = box_length

        self.screening_length = screening_length
        self.include_screening = include_screening

        # Constants
        self.qe = 1.602E-19
        self.eps0 = 8.85E-12

        if efield_data is not None:
            self.Ex_interpolator = UnivariateSpline(grid_data, efield_data, k=spline_order_x, s=smoothing, ext=3)

    def coordinate_transformation(self, r):
        x, y = r2xy(r)
        y_new = self.map_y_into_domain(y)
        r_new = xy2r(x, y_new)
        return r_new

    def map_y_into_domain(self, y, ybounds=None):
        if ybounds is None:
            ybounds = (-self.box_y_length / 2, self.box_y_length / 2)
        return ybounds[0] + (y - ybounds[0]) % (ybounds[1] - ybounds[0])

    def calculate_metrics(self, xi, yi):
        Xi, Yi = np.meshgrid(xi, yi)
        Xj, Yj = Xi.T, Yi.T

        Yi_shifted = Yi.copy()
        Yi_shifted[Yi_shifted > 0] -= self.box_y_length  # Shift entire box length
        Yj_shifted = Yi_shifted.T
        XiXj = Xi - Xj
        YiYj = Yi - Yj
        YiYj_shifted = Yi_shifted - Yj_shifted

        Rij_standard = np.sqrt((XiXj) ** 2 + (YiYj) ** 2)
        Rij_shifted = np.sqrt((XiXj) ** 2 + (YiYj_shifted) ** 2)
        Rij = np.minimum(Rij_standard, Rij_shifted)

        # Use shifted y-coordinate only in this case:
        np.copyto(YiYj, YiYj_shifted, where=Rij_shifted < Rij_standard)

        return XiXj, YiYj, Rij

    def Ex(self, xi, yi):
        return self.Ex_interpolator(xi)

    def V(self, xi, yi):
        return self.interpolator(xi)

    def Velectrostatic(self, xi, yi):
        return self.qe * np.sum(self.V(xi, yi))

    def Vee(self, xi, yi, eps=1E-15):
        yi = self.map_y_into_domain(yi)
        XiXj, YiYj, Rij = self.calculate_metrics(xi, yi)
        np.fill_diagonal(Rij, eps)

        if self.include_screening:
            return + 1 / 2. * self.qe ** 2 / (4 * np.pi * self.eps0) * np.exp(-Rij / self.screening_length) / Rij
        else:
            return + 1 / 2. * self.qe ** 2 / (4 * np.pi * self.eps0) * 1 / Rij

    def Vtotal(self, r):
        xi, yi = r[::2], r[1::2]
        Vtot = self.Velectrostatic(xi, yi)
        interaction_matrix = self.Vee(xi, yi)
        np.fill_diagonal(interaction_matrix, 0)
        Vtot += np.sum(interaction_matrix)
        return Vtot / self.qe

    def dVdx(self, xi, yi):
        return self.derivative(xi)

    def ddVdx(self, xi, yi):
        return self.second_derivative(xi)

    def dVdy(self, xi, yi):
        return np.zeros(len(xi))

    def ddVdy(self, xi, yi):
        return np.zeros(len(xi))

    def grad_Vee(self, xi, yi, eps=1E-15):
        yi = self.map_y_into_domain(yi)

        XiXj, YiYj, Rij = self.calculate_metrics(xi ,yi)
        np.fill_diagonal(Rij, eps)

        gradx_matrix = np.zeros(np.shape(Rij))
        grady_matrix = np.zeros(np.shape(Rij))
        gradient = np.zeros(2 * len(xi))

        if self.include_screening:
            gradx_matrix = -1 * self.qe ** 2 / (4 * np.pi * self.eps0) * np.exp(-Rij/self.screening_length) * \
                           XiXj * (Rij + self.screening_length) / (self.screening_length * Rij ** 3)
            grady_matrix = +1 * self.qe ** 2 / (4 * np.pi * self.eps0) * np.exp(-Rij/self.screening_length) * \
                           YiYj * (Rij + self.screening_length) / (self.screening_length * Rij ** 3)
        else:
            gradx_matrix = -1 * self.qe ** 2 / (4 * np.pi * self.eps0) * XiXj / Rij ** 3
            grady_matrix = +1 * self.qe ** 2 / (4 * np.pi * self.eps0) * YiYj / Rij ** 3

        np.fill_diagonal(gradx_matrix, 0)
        np.fill_diagonal(grady_matrix, 0)

        gradient[::2] = np.sum(gradx_matrix, axis=0)
        gradient[1::2] = np.sum(grady_matrix, axis=0)

        return gradient

    def grad_total(self, r):
        """
        Total derivative of the cost function. This may be used in the optimizer to converge faster.
        :param r: r = np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
        :return: 1D array of length len(r)
        """
        xi, yi = r[::2], r[1::2]
        gradient = np.zeros(len(r))
        gradient[::2] = self.dVdx(xi, yi)
        gradient[1::2] = self.dVdy(xi, yi)
        gradient += self.grad_Vee(xi, yi) / self.qe
        return gradient

    def thermal_kick_x(self, x, y, T):
        kB = 1.38E-23
        qe = 1.602E-19
        ktrapx = np.abs(qe * self.ddVdx(x, y))
        return np.sqrt(2 * kB * T / ktrapx)

    def single_thread(self, iteration, electron_initial_positions, T, cost_function, minimizer_dict):
        xi, yi = r2xy(electron_initial_positions)
        np.random.seed(np.int(time.time()) + iteration)
        xi_prime = xi + self.thermal_kick_x(xi, yi, T) * np.random.randn(len(xi))
        yi_prime = yi + self.thermal_kick_x(xi, yi, T) * np.random.randn(len(yi))
        electron_perturbed_positions = xy2r(xi_prime, yi_prime)
        return minimize(cost_function, electron_perturbed_positions, **minimizer_dict)

    def parallel_perturb_and_solve(self, cost_function, N_perturbations, T, solution_data_reference, minimizer_dict):
        """
        This function is to be run after a minimization by scipy.optimize.minimize has already occured.
        It takes the output of that function in solution_data_reference and tries to find a lower energy state
        by perturbing the system N_perturbation times at temperature T. See thermal_kick_x and thermal_kick_y.
        This function runs N_perturbations in parallel on the cores of your CPU.
        :param cost_function: A function that takes the electron positions and returns the total energy
        :param N_perturbations: Integer, number of perturbations to find a new minimum
        :param T: Temperature to perturb the system at. This is used to convert to a motion.
        :param solution_data_reference: output of scipy.optimize.minimize
        :param minimizer_dict: Dictionary with optimizer options. See scipy.optimize.minimize
        :return: output of minimize with the lowest evaluated cost function
        """
        electron_initial_positions = solution_data_reference['x']
        best_result = solution_data_reference
        pool = multiprocessing.Pool()

        tasks = []
        iteration = 0
        while iteration < N_perturbations:
            iteration += 1
            tasks.append((iteration, electron_initial_positions, T, cost_function, minimizer_dict,))

        results = [pool.apply_async(self.single_thread, t) for t in tasks]
        for result in results:
            res = result.get()

            if res['status'] == 0 and res['fun'] < best_result['fun']:
                cprint("\tNew minimum was found after perturbing!", "green")
                best_result = res
            elif res['status'] == 0 and res['fun'] > best_result['fun']:
                pass  # No new minimum was found after perturbation, this is quite common.
            elif res['status'] != 0 and res['fun'] < best_result['fun']:
                cprint("\tThere is a lower state, but minimizer didn't converge!", "red")
            elif res['status'] != 0 and res['fun'] > best_result['fun']:
                cprint("\tMinimizer didn't converge, but this is not the lowest energy state!", "magenta")

        return best_result

    def sequential_perturb_and_solve(self, cost_function, N_perturbations, T, solution_data_reference, minimizer_options):
        """
        This function is to be run after a minimization by scipy.optimize.minimize has already occured.
        It takes the output of that function in solution_data_reference and tries to find a lower energy state
        by perturbing the system N_perturbation times at temperature T. See thermal_kick_x and thermal_kick_y.
        :param cost_function: A function that takes the electron positions and returns the total energy
        :param N_perturbations: Integer, number of perturbations to find a new minimum
        :param T: Temperature to perturb the system at. This is used to convert to a motion.
        :param solution_data_reference: output of scipy.optimize.minimize
        :param minimizer_options: Dictionary with optimizer options. See scipy.optimize.minimize
        :return: output of minimize with the lowest evaluated cost function
        """

        electron_initial_positions = solution_data_reference['x']
        best_result = solution_data_reference

        for n in range(N_perturbations):
            xi, yi = r2xy(electron_initial_positions)
            xi_prime = xi + self.thermal_kick_x(xi, yi, T) * np.random.randn(len(xi))
            yi_prime = yi + self.thermal_kick_x(xi, yi, T) * np.random.randn(len(yi))
            electron_perturbed_positions = xy2r(xi_prime, yi_prime)

            res = minimize(cost_function, electron_perturbed_positions, **minimizer_options)

            if res['status'] == 0 and res['fun'] < best_result['fun']:
                cprint("\tNew minimum was found after perturbing!", "green")
                best_result = res
            elif res['status'] == 0 and res['fun'] > best_result['fun']:
                pass  # No new minimum was found after perturbation, this is quite common.
            elif res['status'] != 0 and res['fun'] < best_result['fun']:
                cprint("\tThere is a lower state, but minimizer didn't converge!", "red")
            elif res['status'] != 0 and res['fun'] > best_result['fun']:
                cprint("\tMinimizer didn't converge, but this is not the lowest energy state!", "magenta")

        return best_result

    def draw_resonator_pins(self, pin_width, center_gap, edge_gap, **plot_kwargs):
        box_y_length = self.box_y_length*1E6

        pin1_x = [-(center_gap/2.+pin_width), -center_gap/2., -center_gap/2., -(center_gap/2.+pin_width),
                  -(center_gap/2.+pin_width)]
        pin1_y = [-box_y_length/2., -box_y_length/2., box_y_length/2., box_y_length/2., -box_y_length/2.]

        pin2_x = [(center_gap / 2. + pin_width), center_gap / 2., center_gap / 2., (center_gap / 2. + pin_width),
                  (center_gap / 2. + pin_width)]
        pin2_y = [-box_y_length / 2., -box_y_length / 2., box_y_length / 2., box_y_length / 2., -box_y_length / 2.]

        plt.plot(pin1_x, pin1_y, **plot_kwargs)
        plt.plot(pin2_x, pin2_y, **plot_kwargs)
        plt.plot([-(edge_gap + pin_width + center_gap/2.), -(edge_gap + pin_width + center_gap/2.)],
                 [-box_y_length/2., box_y_length/2.], **plot_kwargs)
        plt.plot([(edge_gap + pin_width + center_gap / 2.), (edge_gap + pin_width + center_gap / 2.)],
                 [-box_y_length / 2., box_y_length / 2.], **plot_kwargs)

class CombinedModelSolver:

    def __init__(self, grid_data_x, grid_data_y, potential_data, resonator_electron_configuration,
                 spline_order_x=3, spline_order_y=3, smoothing=0):

        self.interpolator = RectBivariateSpline(grid_data_x, grid_data_y, potential_data,
                                                kx=spline_order_x, ky=spline_order_y, s=smoothing)

        # Constants
        self.qe = 1.602E-19
        self.eps0 = 8.85E-12
        self.kB = 1.38E-23

        # Coordinates for the background potential due to electrons on the resonator
        x_res, y_res = r2xy(resonator_electron_configuration)
        self.x_res = x_res
        self.y_res = y_res

    def V(self, xi, yi):
        """
        Evaluate the electrostatic potential at coordinates xi, yi
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return: Interpolated value(s) of the data supplied to __init__ at values (xi, yi)
        """
        return self.interpolator.ev(xi, yi)

    def Velectrostatic(self, xi, yi):
        """
        When supplying two arrays of size n to V, it returns an array
        of size nxn, according to the meshgrid it has evaluated. We're only interested
        in the sum of the diagonal elements, so we take the sum and this represents
        the sum of the static energy of the n particles in the potential.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        """
        return self.qe * np.sum(self.V(xi, yi))

    def Vbg(self, xi, yi, eps=1E-15):
        """
        Background potential due to electrons on the resonator
        :param xi: a 1D array or float
        :param yi: a 1D array or float (must be same length as xi)
        :return: an array of length xi containing the background potential evaluated at points (xi, yi)
        """
        X_res, X_i = np.meshgrid(self.x_res, xi)
        Y_res, Y_i = np.meshgrid(self.y_res, yi)

        Rij = np.sqrt((X_i - X_res) ** 2 + (Y_i - Y_res) ** 2)
        V = self.qe ** 2 / (4 * np.pi * self.eps0) * 1 / Rij
        return np.sum(V, axis=1)

    def Vee(self, xi, yi, eps=1E-15):
        """
        Returns the repulsion potential between two electrons separated by a distance sqrt(|xi-xj|**2 + |yi-yj|**2)
        Note the factor 1/2. in front of the potential energy to avoid overcounting.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        """
        XiXj, YiYj, Rij = self.calculate_metrics(xi, yi)
        np.fill_diagonal(Rij, eps)
        return + 1 / 2. * self.qe ** 2 / (4 * np.pi * self.eps0) * 1 / Rij

    def Vtotal(self, r):
        """
        This can be used as a cost function for the optimizer.
        Returns the total energy of N electrons
        r is a 0D array with coordinates of the electrons.
        The x-coordinates are thus given by the even elements of r: r[::2],
        whereas the y-coordinates are the odd ones: r[1::2]
        :param r: r = np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
        :return: Scalar with the total energy of the system.
        """
        xi, yi = r[::2], r[1::2]
        Vtot = self.Velectrostatic(xi, yi) + np.sum(self.Vbg(xi, yi))
        interaction_matrix = self.Vee(xi, yi)
        np.fill_diagonal(interaction_matrix, 0)
        Vtot += np.sum(interaction_matrix)
        return Vtot / self.qe

    def dVbgdx(self, xi, yi, eps=1E-15):
        """
        Derivative of the back ground potential due to the resonator electrons
        :param xi: a 1D array or float
        :param yi: a 1D array or float (must be same length as xi)
        :return: an array of length xi containing dV_bg/dx evaluated at points (xi, yi)
        """
        X_res, X_i = np.meshgrid(self.x_res, xi)
        Y_res, Y_i = np.meshgrid(self.y_res, yi)

        Rij = np.sqrt((X_i - X_res) ** 2 + (Y_i - Y_res) ** 2)
        dVdx = - self.qe / (4 * np.pi * self.eps0) * (X_i - X_res) / Rij ** 3
        return np.sum(dVdx, axis=1)

    def dVbgdy(self, xi, yi, eps=1E-15):
        """
        Derivative of the back ground potential due to the resonator electrons
        :param xi: a 1D array or float
        :param yi: a 1D array or float (must be same length as xi)
        :return: an array of length xi containing dV_bg/dy evaluated at points (xi, yi)
        """
        X_res, X_i = np.meshgrid(self.x_res, xi)
        Y_res, Y_i = np.meshgrid(self.y_res, yi)

        Rij = np.sqrt((X_i - X_res) ** 2 + (Y_i - Y_res) ** 2)
        dVdy = - self.qe / (4 * np.pi * self.eps0) * (Y_i - Y_res) / Rij ** 3
        return np.sum(dVdy, axis=1)

    def dVdx(self, xi, yi):
        """
        Derivative of the electrostatic potential in the x-direction.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=1, dy=0)

    def ddVdx(self, xi, yi):
        """
        Second derivative of the electrostatic potential in the x-direction.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=2, dy=0)

    def dVdy(self, xi, yi):
        """
        Derivative of the electrostatic potential in the y-direction
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=0, dy=1)

    def ddVdy(self, xi, yi):
        """
        Second derivative of the electrostatic potential in the y-direction.
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :return:
        """
        return self.interpolator.ev(xi, yi, dx=0, dy=2)

    def grad_Vee(self, xi, yi, eps=1E-15):
        """
        Derivative of the electron-electron interaction term
        :param xi: a 1D array, or float
        :param yi: a 1D array or float
        :param eps: A small but non-zero number to avoid triggering Warning message. Exact value is irrelevant.
        :return: 1D-array of size(xi) + size(yi)
        """
        # Resonator method
        XiXj, YiYj, Rij = self.calculate_metrics(xi, yi)
        np.fill_diagonal(Rij, eps)

        gradx_matrix = np.zeros(np.shape(Rij))
        grady_matrix = np.zeros(np.shape(Rij))
        gradient = np.zeros(2 * len(xi))

        gradx_matrix = -1 * self.qe ** 2 / (4 * np.pi * self.eps0) * (XiXj) / Rij ** 3
        np.fill_diagonal(gradx_matrix, 0)

        grady_matrix = +1 * self.qe ** 2 / (4 * np.pi * self.eps0) * (YiYj) / Rij ** 3
        np.fill_diagonal(grady_matrix, 0)

        gradient[::2] = np.sum(gradx_matrix, axis=0)
        gradient[1::2] = np.sum(grady_matrix, axis=0)

        return gradient

    def grad_total(self, r):
        """
        Total derivative of the cost function. This may be used in the optimizer to converge faster.
        :param r: r = np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
        :return: 1D array of length len(r)
        """
        xi, yi = r[::2], r[1::2]
        gradient = np.zeros(len(r))
        gradient[::2] = self.dVdx(xi, yi) + self.dVbgdx(xi, yi)
        gradient[1::2] = self.dVdy(xi, yi) + self.dVbgdy(xi, yi)
        gradient += self.grad_Vee(xi, yi) / self.qe
        return gradient

    def calculate_metrics(self, xi, yi):
        """
        Calculate the pairwise distance between electron combinations (xi, yi)
        This is used for calculating grad_Vee for example.
        :param xi: a 1D array, or float
        :param yi: a 1D array, or float
        :return: Xi - Xj, Yi - Yj, Rij = sqrt((Xi-Xj)^2 + (Yi-Yj)^2)
        """
        Xi, Yi = np.meshgrid(xi, yi)
        Xj, Yj = Xi.T, Yi.T

        Rij = np.sqrt((Xi - Xj) ** 2 + (Yi - Yj) ** 2)

        XiXj = Xi - Xj
        YiYj = Yi - Yj
        return XiXj, YiYj, Rij

    def thermal_kick_x(self, x, y, T):
        ktrapx = np.abs(self.qe * self.ddVdx(x, y))
        return np.sqrt(2 * self.kB * T / ktrapx)

    def thermal_kick_y(self, x, y, T):
        ktrapy = np.abs(self.qe * self.ddVdy(x, y))
        return np.sqrt(2 * self.kB * T / ktrapy)

    def perturb_and_solve(self, cost_function, N_perturbations, T, solution_data_reference, **minimizer_options):
        """
        This function is to be run after a minimization by scipy.optimize.minimize has already occured.
        It takes the output of that function in solution_data_reference and tries to find a lower energy state
        by perturbing the system N_perturbation times at temperature T. See thermal_kick_x and thermal_kick_y.
        :param cost_function: A function that takes the electron positions and returns the total energy
        :param N_perturbations: Integer, number of perturbations to find a new minimum
        :param T: Temperature to perturb the system at. This is used to convert to a motion.
        :param solution_data_reference: output of scipy.optimize.minimize
        :param minimizer_options: Dictionary with optimizer options. See scipy.optimize.minimize
        :return: output of minimize with the lowest evaluated cost function
        """

        electron_initial_positions = solution_data_reference['x']
        best_result = solution_data_reference

        for n in range(N_perturbations):
            xi, yi = r2xy(electron_initial_positions)
            xi_prime = xi + self.thermal_kick_x(xi, yi, T) * np.random.randn(len(xi))
            yi_prime = yi + self.thermal_kick_y(xi, yi, T) * np.random.randn(len(yi))
            electron_perturbed_positions = xy2r(xi_prime, yi_prime)

            res = minimize(cost_function, electron_perturbed_positions, method='CG', **minimizer_options)

            if res['status'] == 0 and res['fun'] < best_result['fun']:
                cprint("\tNew minimum was found after perturbing!", "green")
                best_result = res
            elif res['status'] == 0 and res['fun'] > best_result['fun']:
                pass  # No new minimum was found after perturbation, this is quite common.
            elif res['status'] != 0 and res['fun'] < best_result['fun']:
                cprint("\tThere is a lower state, but minimizer didn't converge!", "red")
            elif res['status'] != 0 and res['fun'] > best_result['fun']:
                cprint("\tMinimizer didn't converge, but this is not the lowest energy state!", "magenta")

        return best_result

######################
## HELPER FUNCTIONS ##
######################

def construct_symmetric_y(ymin, N):
    """
    This helper function constructs a one-sided array from ymin to -dy/2 with N points.
    The spacing is chosen such that, when mirrored around y = 0, the spacing is constant.

    This requirement limits our choice for dy, because the spacing must be such that there's
    an integer number of points in yeval. This can only be the case if
    dy = 2 * ymin / (2*k+1) and Ny = ymin / dy - 0.5 + 1
    yeval = y0, y0 - dy, ... , -3dy/2, -dy/2
    :param ymin: Most negative value
    :param N: Number of samples in the one-sided array
    :return: One-sided array of length N.
    """
    dy = 2 * np.abs(ymin) / np.float(2 * N + 1)
    return np.linspace(ymin, -dy / 2., int((np.abs(ymin) - 0.5 * dy) / dy + 1))


def load_data(data_path, datafiles=None, names=None, xeval=None, yeval=None, mirror_y=True, do_plot=True, extend_resonator=True,
              inserted_trap_length=0, inserted_res_length=40, smoothen_xy=None):
    """
    Takes in the following file names: "Resonator.dsp", "Trap.dsp", "ResonatorGuard.dsp", "CenterGuard.dsp" and
    "TrapGuard.dsp" in path "data_path" and loads the data into a dictionary output.
    :param data_path: Data path that contains the simulation files
    :param xeval: a 1D array. Interpolate the potential data for these x-points. Units: um
    :param yeval: a 1D array. Interpolate the potential data for these y-points. Units: um
    :param mirror_y: bool, mirror the potential data around the y-axis. To use this make sure yeval is symmetric around 0
    :param extend_resonator: bool, extend potential data to the right of the minimum of the resonator potential data
    :param do_plot: Plot the data on the grid made up by xeval and yeval
    :param inserted_trap_length:
    :param inserted_res_length:
    :param smoothen_xy: Smooth the raw data according to a window in the x and y direction window = (x,y).
    The program will calculate the window size for the moving average filter in each direction according to (x, y).
    :return: x, y, output = [{'name' : ..., 'V' : ..., 'x' : ..., 'y' : ...}, ...]
    """
    if datafiles is None:
        datafiles = ["Resonator.dsp", "Trap.dsp", "ResonatorGuard.dsp", #"CenterGuard.dsp",
                     "TrapGuard.dsp"]

    output = list()
    if names is None:
        names = ['resonator', 'trap', 'resonatorguard', #'centerguard',
                 'trapguard']
    idx = 1

    insert_resonator = True if inserted_res_length > 0 else False
    insert_trap = True if inserted_trap_length > 0 else False
    if insert_trap:
        names = ['trap', 'resonator', 'resonatorguard', 'trapguard']
        datafiles = ["Trap.dsp", "Resonator.dsp", "ResonatorGuard.dsp", "TrapGuard.dsp"]


    # Iterate over the data files
    for name, datafile in zip(names, datafiles):
        elements, nodes, elem_solution, bounding_box = load_dsp(os.path.join(data_path, datafile))
        xdata, ydata, Udata = interpolate_slow.prepare_for_interpolation(elements, nodes, elem_solution)
        xcenter = np.mean(bounding_box[0:2])
        yedge = bounding_box[3]
        xdata -= xcenter
        ydata -= yedge

        if xeval is None:
            xeval = xcenter + np.linspace(np.min(xdata), np.max(xdata), 101)
        if yeval is None:
            yeval = yedge + np.linspace(np.min(ydata), np.max(ydata), 101)

        xinterp, yinterp, Uinterp = interpolate_slow.evaluate_on_grid(xdata, ydata, Udata, xeval=xeval, yeval=yeval,
                                                                      clim=(0.0, 1.0), plot_axes='xy',
                                                                      cmap=plt.cm.Spectral_r, plot_mesh=False,
                                                                      plot_data=False)

        if smoothen_xy is not None:
            dx = np.diff(xeval)[0] * 1E-6
            dy = np.diff(yeval)[0] * 1E-6
            Nrows = np.int(smoothen_xy[0]/dx)
            Ncols = np.int(smoothen_xy[1]/dy)

            if Nrows > 1 or Ncols > 1:
                Uinterp = common.moving_average_2d(Uinterp, (Nrows, Ncols))
            elif Nrows <= 1:
                print("Smoothing has no effect in y-direction, please increase sampling in y direction.")
            elif Ncols <= 1:
                print("Smoothing has no effect in x-direction, please increase sampling in x direction.")

        if mirror_y:
            # Mirror around the y-axis
            ysize, xsize = np.shape(Uinterp)
            Uinterp_symmetric = np.zeros((2 * ysize, xsize))
            Uinterp_symmetric[:ysize, :] = Uinterp
            Uinterp_symmetric[ysize:, :] = Uinterp[::-1, :]

            y_symmetric = np.zeros((2 * ysize, xsize))
            y_symmetric[:ysize, :] = yinterp
            y_symmetric[ysize:, :] = -yinterp[::-1, :]

            x_symmetric = np.zeros((2 * ysize, xsize))
            x_symmetric[:ysize, :] = xinterp
            x_symmetric[ysize:, :] = xinterp

            #yeval = np.append(yeval, -yeval[::-1])
        else:
            x_symmetric = xinterp
            y_symmetric = yinterp
            Uinterp_symmetric = Uinterp

        if extend_resonator:
            # Draw an imaginary line at the minimum of the resonator potential and
            # substitute the data to the right of that line with the minimum value
            if name == "resonator":
                res_left_idx = np.argmax(Uinterp_symmetric[common.find_nearest(y_symmetric[:, 0], 0), :])
                res_left_x = x_symmetric[0, res_left_idx]
                res_right_x = x_symmetric[0, -1]
                # Note: res_left_x and res_right_x are in units of um
                print("Resonator data was replaced from x = %.2f um to x = %.2f um" % (res_left_x, res_right_x))

                for i in range(res_left_idx, np.shape(Uinterp_symmetric)[1]):
                    Uinterp_symmetric[:, i] = Uinterp_symmetric[:, res_left_idx]
            else:
                # Also change all the other potential data
                for i in range(res_left_idx, np.shape(Uinterp_symmetric)[1]):
                    Uinterp_symmetric[:, i] = Uinterp_symmetric[:, res_left_idx]

        elif insert_resonator:
            # Insertion is done at the minimum of the potential.
            inserted_length = inserted_res_length
            x_spacing = x_symmetric[0, 1] - x_symmetric[0, 0]
            new_indices = np.int(inserted_length/x_spacing)
            old_shape = np.shape(Uinterp_symmetric)
            new_shape = (old_shape[0], old_shape[1] + new_indices)

            #print("We start with a %d by %d array and extend it to a %d by %d array"%(old_shape[0], old_shape[1], new_shape[0], new_shape[1]))

            Uinterp_symmetric_new = np.zeros(new_shape)

            if name == "resonator":
                res_left_idx = np.argmax(Uinterp_symmetric[common.find_nearest(y_symmetric[:, 0], 0), :])
                res_left_x = x_symmetric[0, res_left_idx]
                res_right_idx = res_left_idx + new_indices
                res_right_x = res_left_x + new_indices * x_spacing

                # Note: res_left_x and res_right_x are in units of um
                print("Resonator data was inserted from x = %.2f um to x = %.2f um" % (res_left_x, res_right_x))

            for i in range(res_left_idx, res_right_idx):
                Uinterp_symmetric_new[:, i] = Uinterp_symmetric[:, res_left_idx]

            Uinterp_symmetric_new[:, :res_left_idx] = Uinterp_symmetric[:, :res_left_idx]
            Uinterp_symmetric_new[:, res_right_idx:] = Uinterp_symmetric[:, res_left_idx:]
            xs = np.arange(np.min(x_symmetric), np.max(x_symmetric) + (new_indices) * x_spacing, x_spacing)

            if len(xs) != new_shape[1]:
                print("xs shapes are not correct: %d vs. %d"%(len(xs), new_shape[1]))

            ys = np.linspace(np.min(y_symmetric), np.max(y_symmetric), new_shape[0])

            if len(ys) != new_shape[0]:
                print("ys shapes are not correct: %d vs. %d" % (len(ys), new_shape[0]))

            x_symmetric_new, y_symmetric_new = np.meshgrid(xs, ys)
            x_symmetric, y_symmetric = x_symmetric_new, y_symmetric_new
            Uinterp_symmetric = Uinterp_symmetric_new

        elif insert_trap:
            # Insertion is done at the minimum of the potential.
            inserted_length = inserted_trap_length
            x_spacing = x_symmetric[0, 1] - x_symmetric[0, 0]
            new_indices = np.int(inserted_length / x_spacing)
            old_shape = np.shape(Uinterp_symmetric)
            new_shape = (old_shape[0], old_shape[1] + new_indices)

            # print("We start with a %d by %d array and extend it to a %d by %d array"%(old_shape[0], old_shape[1],
            # new_shape[0], new_shape[1]))

            Uinterp_symmetric_new = np.zeros(new_shape)

            if name == "trap":
                trap_left_idx = np.argmax(Uinterp_symmetric[common.find_nearest(y_symmetric[:, 0], 0), :])
                trap_left_x = x_symmetric[0, trap_left_idx]
                trap_right_idx = trap_left_idx + new_indices
                trap_right_x = trap_left_x + new_indices * x_spacing

                # Note: res_left_x and res_right_x are in units of um
                print("Trap data was inserted from x = %.2f um to x = %.2f um" % (trap_left_x, trap_right_x))

            for i in range(trap_left_idx, trap_right_idx):
                Uinterp_symmetric_new[:, i] = Uinterp_symmetric[:, trap_left_idx]

            Uinterp_symmetric_new[:, :trap_left_idx] = Uinterp_symmetric[:, :trap_left_idx]
            Uinterp_symmetric_new[:, trap_right_idx:] = Uinterp_symmetric[:, trap_left_idx:]
            xs = np.arange(np.min(x_symmetric), np.max(x_symmetric) + (new_indices) * x_spacing, x_spacing)

            if len(xs) != new_shape[1]:
                print("xs shapes are not correct: %d vs. %d" % (len(xs), new_shape[1]))

            ys = np.linspace(np.min(y_symmetric), np.max(y_symmetric), new_shape[0])

            if len(ys) != new_shape[0]:
                print("ys shapes are not correct: %d vs. %d" % (len(ys), new_shape[0]))

            x_symmetric_new, y_symmetric_new = np.meshgrid(xs, ys)
            x_symmetric, y_symmetric = x_symmetric_new, y_symmetric_new
            Uinterp_symmetric = Uinterp_symmetric_new

        if do_plot:
            plt.figure(figsize=(8., 2.))
            common.configure_axes(12)
            plt.title(name)
            plt.pcolormesh(x_symmetric, y_symmetric, Uinterp_symmetric, cmap=plt.cm.Spectral_r, vmin=0.0, vmax=1.0)
            plt.colorbar()
            plt.xlim(np.min(x_symmetric), np.max(x_symmetric))
            plt.ylim(np.min(y_symmetric), np.max(y_symmetric))
            plt.ylabel("$y$ ($\mu$m)")

        output.append({'name': name,
                       'V': np.array(Uinterp_symmetric.T, dtype=np.float64),
                       'x': np.array(x_symmetric.T, dtype=np.float64),
                       'y': np.array(y_symmetric.T, dtype=np.float64)})

        idx += 1

    if insert_trap:
        output[0], output[1] = output[1], output[0]

    return x_symmetric*1E-6, y_symmetric*1E-6, output

def draw_from_dxf(filename, ax=None, offset=(0E-6, 0E-6), fill=False, fill_colors=None, **plot_options):
    """
    Draws polylines from a dxf file into a graph
    :param filename: dxf file name.
    :param plot_options: dictionary of plot attributes, e.g. {'color':'k'}
    :return: Nones
    """
    dxf = dxfgrabber.readfile(filename)
    output = [entity for entity in dxf.entities]
    i = 0
    for o in output:
        if o.dxftype == "LWPOLYLINE" or o.dxftype == "POLYLINE":
            r = np.array(o.points)
            x = r[:,0] + offset[0]
            y = r[:,1] + offset[1]
            if fill:
                if fill_colors is None:
                    fill_color = 'gray'
                else:
                    fill_color = fill_colors[i]
                if ax is None:
                    plt.fill(x, y, color=fill_color, **plot_options)
                else:
                    ax.fill(x, y, color=fill_color, **plot_options)
                i += 1
            else:
                if ax is None:
                    plt.plot(x, y, **plot_options)
                else:
                    ax.plot(x, y, **plot_options)

def factors(n):
    """
    Get integer divisors of an integer n
    :param n: integer to get the factors from
    :return: A list of divisors of n
    """
    return list(functools.reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def is_prime(n):
    """
    Checks if a number is a prime number.
    :param n: Integer to check
    :return: True if n is a prime, False if not
    """
    divisors = factors(n)
    divisors.remove(n)
    if n != 1:
        divisors.remove(1)
    return divisors == list()

def get_rectangular_initial_condition(N_electrons, N_rows=None, N_cols=None,
                                      x0=0E-6, y0=0E-6, dx=0.20E-6, dy=0.40E-6):
    """
    Note: N_rows * N_cols must be N_electrons
    :param N_electrons: Number of electrons
    :param N_rows: Number of rows
    :param N_cols: Number of columns
    :return: Rectangular electron configuration where r = [x0, y0, x1, y1, ...]
    """
    add_electrons_later = 0
    if (N_rows is None) or (N_cols is None):
        divisors = factors(N_electrons)
        divisors.remove(N_electrons)
        divisors.remove(1)

        if divisors == list():
            # Catch a large prime number and avoid it being one long row.
            while is_prime(N_electrons):
                add_electrons_later += 1
                N_electrons -= 1

            divisors = factors(N_electrons)
            divisors.remove(N_electrons)
            divisors.remove(1)

        optimal_index = np.argmin(np.abs(np.array(divisors)-np.sqrt(N_electrons)))
        N_rows = np.int(divisors[optimal_index])
        N_cols = np.int(N_electrons/N_rows)

    if N_cols * N_rows != N_electrons:
        raise ValueError("N_cols and N_rows are not compatible with N_electrons")
    else:
        y_separation = dy
        x_separation = dx
        ys = np.linspace(y0 - (N_cols - 1) / 2. * y_separation, y0 + (N_cols - 1) / 2. * y_separation, N_cols)
        yinit = np.tile(np.array(ys), N_rows)
        xs = np.linspace(x0 - (N_rows - 1) / 2. * x_separation, x0 + (N_rows - 1) / 2. * x_separation, N_rows)
        xinit = np.repeat(xs, N_cols)

    for k in range(add_electrons_later):
        xinit = np.append(xinit, [x0 + (N_rows + 1) / 2. * x_separation])
        yinit = np.append(yinit, [y0 - (N_cols - 1 - 2 * k) / 2. * y_separation])

    return xy2r(xinit, yinit)

def check_unbounded_electrons(ri, xdomain, ydomain):
    """
    This helper function checks if any electrons have escaped the bounds of the simulation box defined by xdomain and
    ydomain. It returns the number of electrons outside the rectangular domain.
    :param ri: electron positions
    :param xdomain: (xmin, xmax)
    :param ydomain: (ymin, ymax)
    :return: Number of electrons outside domain
    """
    xleft, xright = xdomain
    ybottom, ytop = ydomain

    xi, yi = r2xy(ri)

    questionable_electrons = list()

    questionable = np.where(xi < xleft)[0]
    for q in questionable:
        questionable_electrons.append(q)
    questionable = np.where(xi > xright)[0]
    for q in questionable:
        questionable_electrons.append(q)
    questionable = np.where(yi < ybottom)[0]
    for q in questionable:
        questionable_electrons.append(q)
    questionable = np.where(yi > ytop)[0]
    for q in questionable:
        questionable_electrons.append(q)

    return len(np.unique(np.array(questionable_electrons)))

def setup_initial_condition(N_electrons, xdomain, ydomain, x0, y0, dx=None, dy=None):
    """
    This helper function takes away some of the manual labour involved in setting up the initial condition.
    It sets up a grid of electrons centered at (x0, y0) with spacings dx and dy (optional arguments).
    It will reduce the spacing dx and dy until all the electrons are within the rectangular domain specified by
    xdomain and ydomain
    :param N_electrons: Number of electrons
    :param xdomain: (xmin, xmax)
    :param ydomain: (ymin, ymax)
    :param x0: x-coordinate of the center of the configuration
    :param y0: y-coordinate of the center of the configuration
    :param dx: Spacing in the x-direction (optional)
    :param dy: Spacing in the y-direction (optional)
    :return: Electron configuration
    """
    if dx is None:
        dx = 0.50E-6
    if dy is None:
        dy = 0.50E-6

    N_electrons_outside = N_electrons

    if check_unbounded_electrons(xy2r([x0], [y0]), xdomain, ydomain):
        raise ValueError("Center of initial condition lies outside domain!")

    while N_electrons_outside > 0:
        r_init = get_rectangular_initial_condition(N_electrons, N_rows=None, N_cols=None, x0=x0, y0=y0, dx=dx, dy=dy)
        N_electrons_outside = check_unbounded_electrons(r_init, xdomain, ydomain)
        dx /= 2
        dy /= 2

    return r_init

def get_electron_density_by_position(ri):
    """
    Calculate the electron density based on the nearest neighbor for each electron. The electron density is in m^-2
    :param ri: Electron x,y coordinate pairs
    :return: Electron density in m^-2
    """
    xi, yi = r2xy(ri)
    Xi, Yi = np.meshgrid(xi, yi)
    Xj, Yj = Xi.T, Yi.T

    Rij = np.sqrt((Xi - Xj) ** 2 + (Yi - Yj) ** 2)
    np.fill_diagonal(Rij, 1E10)

    nearest_neighbors = np.min(Rij, axis=1)
    ns = 1 / (np.mean(nearest_neighbors)) ** 2
    return ns

def get_electron_density_by_area(ri):
    """
    Calculate the electron density based on the smallest bounding box that can be drawn around the electron configuration
    :param ri: Electron x,y coordinate pairs
    :return: Electron density in m^-2
    """
    xi, yi = r2xy(ri)
    xmin = np.min(xi)
    xmax = np.max(xi)
    ymin = np.min(yi)
    ymax = np.max(yi)
    return len(xi)/(np.abs(xmax-xmin)*(ymax-ymin))

def mirror_pt(p, axis_angle, axis_pt):
    """
    Mirrors point p about a line at angle "axis_angle" intercepting point "axis_pt"
    :param p:
    :param axis_angle:
    :param axis_pt:
    :return:
    """
    theta = axis_angle * np.pi / 180.
    return (axis_pt[0] + (-axis_pt[0] + p[0]) * np.cos(2 * theta) + (-axis_pt[1] + p[1]) * np.sin(2 * theta),
            p[1] + 2 * (axis_pt[1] - p[1]) * np.cos(theta) ** 2 + (-axis_pt[0] + p[0]) * np.sin(2 * theta))

def mirror_pts(points, axis_angle, axis_pt):
    """
    Mirrors an array of points one by one using mirror_pt
    :param points:
    :param axis_angle:
    :param axis_pt:
    :return:
    """
    return [mirror_pt(p, axis_angle, axis_pt) for p in points]

def draw_electrode_outline(filename, x0=None, y0=None, **plot_kwargs):
    """
    Draw an outline of the electrodes from a dxf file.
    :param filename: full path and filename of the dxf file
    :param x0: x-coordinate of the origin of the plot (optional)
    :param y0: y-coordinate of the origin of the plot (optional)
    :param plot_kwargs: optional arguments for the plot, such as linewidth or color
    :return:
    """
    dwg = ezdxf.readfile(filename)
    modelspace = dwg.modelspace()

    if x0 is None:
        x0 = 0
    if y0 is None:
        y0 = 0

    for e in list(modelspace):
        if e.dxftype() == "LWPOLYLINE":
            with e.points() as points:
                x, y = list(), list()
                for p in points:
                    x.append(p[0] - x0)
                    y.append(p[1] - y0)

            points = zip(x, y)
            mirrored_xy = mirror_pts(points, 0, (x0, y0))
            mirrored_x, mirrored_y = zip(*mirrored_xy)

            plt.plot(x, y, **plot_kwargs)
            plt.plot(mirrored_x, mirrored_y, **plot_kwargs)

def r2xy(r):
    """
    Reformat electron position array.
    :param r: r = np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
    :return: np.array([x0, x1, ...]), np.array([y0, y1, ...])
    """
    return r[::2], r[1::2]

def xy2r(x, y):
    """
    Reformat electron position array.
    :param x: np.array([x0, x1, ...])
    :param y: np.array([y0, y1, ...])
    :return: r = np.array([x0, y0, x1, y1, x2, y2, ... , xN, yN])
    """
    if len(x) == len(y):
        r = np.zeros(2 * len(x))
        r[::2] = x
        r[1::2] = y
        return r
    else:
        raise ValueError("x and y must have the same length!")

def map_into_domain(xi, yi, xbounds=(-1,1), ybounds=(-1,1)):
    """
    If electrons leave the simulation box, it may be desirable to map them back into the domain. If the electrons cross
    the right boundary, they will be mirrored and their mapped position is returned.
    :param xi: np.array([x0, x1, ...])
    :param yi: np.array([y0, y1, ...])
    :param xbounds: Tuple specifying the bounds in the x-direction
    :param ybounds: Tuple specifying the bounds in the y-direction
    :return: xi, yi (all within the simulation box)
    """
    left_boundary, right_boundary = xbounds
    bottom_boundary, top_boundary = ybounds

    L = right_boundary - left_boundary
    W = top_boundary - bottom_boundary

    xi = np.abs(L - (xi - right_boundary) % (2 * L)) + left_boundary
    yi[yi>top_boundary] = top_boundary
    yi[yi<bottom_boundary] = bottom_boundary

    return xi, yi
