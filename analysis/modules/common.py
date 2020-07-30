import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import cmath, csv, os
from tabulate import tabulate
from scipy.signal import convolve2d

def rgb(i, N, cmap=plt.cm.plasma):
    """
    Returns rgb-values from a discretized colormap. The color is i out of N indices.
    :param i: integer ranging from 0 to N
    :param N: Number of steps to discretize to colormap.
    :param cmap: a matplotlib colormap instance
    :return: rgb tuple.
    """
    cmap = matplotlib.cm.get_cmap(cmap)
    return cmap(i/float(N))

def load_csv(filename, header_length=7, footer_length=2, ncols=3, complex_data=False):
    """
    Load a csv file into a numpy array.
    :param datafile: filename of the .csv file
    :param header_length: number of header rows that are no data
    :param footer_length: number footer rows that are no data
    :param ncols: Number of columns in the data
    :return:
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        len_data = sum(1 for row in reader) - header_length - footer_length

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for k in range(header_length):
            next(reader, None)


        for idx,row in enumerate(reader):
            if idx == 0:
                dtype = np.complex if complex_data else np.float
                data = np.zeros([len_data, ncols], dtype=dtype)
            try:
                for k in range(ncols):
                    if complex_data:
                        a, b = row[k].strip().strip('(').strip(')').split('+')
                        data[idx, k] = np.complex(a) + np.complex(b)
                    else:
                        data[idx, k] = float(row[k])
            except:
                raise(ValueError("Unexpected data format. Error in parsing string!"))

    return data

def csv_to_h5(filename, data_cache, **kwargs):
    """
    Convert a .csv file from the PNAX to a h5 file with the same name
    :param filename: Filename of the file that needs to be converted
    :return:
    """
    data = load_csv(filename, **kwargs)
    h5_filename = filename[:-4]+'.h5'
    if not os.path.isfile(h5_filename):
        print("%s ..."%h5_filename)
        myfile = data_cache.dataCacheProxy(filepath=h5_filename, expInst=None)
        myfile.append("fpts", data[:,0])
        myfile.append("mags", data[:,1])
        myfile.append("phases", data[:,2])
    else:
        print("%s <-- already exists, skipping..."%h5_filename)

def get_phase(array):
    """
    Returns the phase of a complex array.
    :param array: Array filled with complex numbers
    :return: Array
    """
    phase_out = np.zeros([len(array)])
    for idx,k in enumerate(array):
        phase_out[idx] = cmath.phase(k)
    return phase_out

def remove_offset(xdata, ydata, f0=None):
    """
    Removes a constant offset and places ydata at point f0 at 0.
    :param xdata: fpoints.
    :param ydata: ydata.
    :param f0: point at which ydata will be set to 0. if f0=None it will be the center of the array.
    :return: the ydata with the offset removed.
    """
    if f0 is None:
        f0_idx = len(xdata)/2
    else:
        f0_idx = find_nearest(xdata, f0)
    y_f0 = ydata[f0_idx]
    return ydata - y_f0

def remove_slope(xdata, ydata):
    """
    Removes a slope from data using the first and last points of xdata/ydata to determine the slope.
    :param xdata: fpoints.
    :param ydata: ydata.
    :return: ydata with the slope removed.
    """
    return ydata-(ydata[-1]-ydata[0])/(xdata[-1]-xdata[0])*xdata

def recenter_phase(xdata, ydata, f0):
    """
    Removes the slope and phase offset such that the phase is flat and has a 0 at f0.
    :param xdata: fpoints
    :param ydata: phase
    :param f0: Point at which the phase is 0.
    :return:
    """
    phase = remove_slope(xdata, ydata)
    return remove_offset(xdata, phase, f0)

def find_nearest(array, value):
    """
    Finds the nearest value in array. Returns index of array for which this is true.
    """
    idx=(np.abs(array-value)).argmin()
    return np.int(idx)

def moving_average(ydata, window_size):
    """
    Outputs the moving average of a function interval over a window_size. Output is 
    the same size as the input. 
    """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(ydata, window, 'same')

def moving_average_2d(ydata, window_size):
    """
    Convolve the 2D array ydata with a uniform window. This acts as a 2D low pass filter.
    :param ydata: N x M array to apply the window to
    :param window_size: Tuple: (Nrows, Ncols) where Ncols is the number of columns and NRows is the number of rows
    :return: N x M array (ydata convoluted with the window)
    """
    window = np.ones((int(window_size[1]),int(window_size[0])))/float(window_size[0]*window_size[1])
    return convolve2d(ydata, window, mode='same')

def dBm_to_W(Pdbm):
    """
    Convert power in dBm to power in W
    :param Pdbm: Power in dBm.
    :return: Power in W
    """
    return 10**(Pdbm/10.) * 1E-3

def dBm_to_vrms(Pdbm, Z0=50.):
    """
    Convert power in dBm to rms voltage
    :param Pdbm: Power in dBm.
    :param Z0: Characteristic impedance
    :return: Vrms
    """
    Pmw = 10**(Pdbm/10.)
    return np.sqrt(Pmw*Z0/1E3)

def dBm_to_vpp(Pdbm, Z0=50.):
    """
    Convert power in dbm to peak-to-peak voltage given certain characteristic impedance Z0
    :param Pdbm: Power in dBm
    :param Z0: Characteristic impedance
    :return: Vpp
    """
    Pmw = 10**(Pdbm/10.)
    return 2 * np.sqrt(2) * np.sqrt(Pmw*Z0/1E3)

def save_figure(fig, save_path=None, open_explorer=False):
    """
    Saves a figure with handle "fig" in "save_path". save_path does not need to be specified, if not specified
    the function will create a new file in S:\Gerwin\iPython notebooks\Figures under the current date.
    :param fig: Figure handle
    :param save_path: Filename for the file to be saved. May be None
    :param open_explorer: Open a process of windows explorer showing the file, for easy copy & paste into slides
    :return: Nothing
    """
    import subprocess, time, os

    # Check if path is accessible
    if save_path is None:
        base_path = r"S:\Gerwin\iPython notebooks\Figures"
    else:
        base_path = save_path

    date = time.strftime("%Y%m%d")

    # Create a file name
    file_exists = True
    i = 0
    while file_exists:
        idx = str(100000 + i)
        save_path = os.path.join(base_path, "%s_figure_%s.png"%(date, idx[1:]))
        if not os.path.isfile(save_path):
            break
        else:
            i += 1

    if os.path.exists(os.path.split(save_path)[0]):
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

        time.sleep(1)
        if open_explorer:
            subprocess.Popen(r'explorer /select,"%s"'%save_path)

    else:
        print("Desired path %s does not exist."%(save_path))


def mapped_color_plot(xdata, ydata, cmap=plt.cm.viridis, clim=None, scale_type='sequential', log_scaling=False,
                      colorbar=False, **kwarg):
    """
    Plot points in a data set with different color. The value of the color is determined either by the x-value or the
    y-value and can be scaled linearly or logarithmically.
    :param xdata: x-points
    :param ydata: y-points
    :param cmap: plt.cm instance
    :param clim: Tuple (cmin, cmax). Default is None.
    :param scale_type: Either 'x', 'y', 'sequential' or 'external'
    :param log_scaling: Scale data logarithmically
    :return:
    """
    if clim is None:
        if scale_type == 'x':
            if log_scaling:
                vmin, vmax = np.min(np.log10(xdata)), np.max(np.log10(xdata))
            else:
                vmin, vmax = np.min(xdata), np.max(xdata)
        elif scale_type == 'sequential':
            vmin, vmax = 0, len(xdata)-1
        elif scale_type == 'y':
            if log_scaling:
                vmin, vmax = np.min(np.log10(ydata)), np.max(np.log10(ydata))
            else:
                vmin, vmax = np.min(ydata), np.max(ydata)
    else:
        vmin, vmax = clim

    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    #plt.scatter(xdata, ydata, c=)
    idx = 0
    for x, y in zip(xdata, ydata):
        if scale_type == 'x':
            if log_scaling:
                plt.plot(x, y, 'o', color=m.to_rgba(np.log10(x)), **kwarg)
            else:
                plt.plot(x, y, 'o', color=m.to_rgba(x), **kwarg)
        elif scale_type == 'sequential':
            plt.plot(x, y, 'o', color=m.to_rgba(idx), **kwarg)
        else:
            if log_scaling:
                plt.plot(x, y, 'o', color=m.to_rgba(np.log10(y)), **kwarg)
            else:
                plt.plot(x, y, 'o', color=m.to_rgba(y), **kwarg)
        idx+=1

    if colorbar:
        m._A = []
        plt.colorbar(m)

def configure_axes(fontsize):
    """
    Creates axes in the Arial font with fontsize as specified by the user
    :param fontsize: Font size in points, used for the axes labels and ticks.
    :return: None
    """
    import platform
    if platform.system() == 'Linux':
        matplotlib.rc('font', **{'size': fontsize})
        matplotlib.rc('figure', **{'dpi': 80, 'figsize': (6.,4.)})
    else:
        matplotlib.rc('font', **{'size': fontsize, 'family':'sans-serif','sans-serif':['Arial']})
        matplotlib.rc('figure', **{'dpi': 80, 'figsize': (6.,4.)})

def legend_outside(**kwargs):
    """
    Places the legend outside the figure area. Useful for a busy figure where the legend overshadows data.
    :param kwargs: Extra arguments
    :return: None
    """
    ax = plt.gca()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), **kwargs)

    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

def plot_opt(color, style='transparent', msize=8):
    """
    Usage: plot(x, y, **plot_opt('red'))
    :param color: String denoting the color of the plot
    :param style: Select 'transparent' for transparent plotting style.
    :param msize: Marker size in points. Defaults to 8 if not specified.
    :return: Dictionary of marker styles.
    """
    if style == 'transparent':
        return {'marker': 'o', 'ms':msize, 'mew':1, 'mfc': color ,'mec': color, 'alpha' : 0.4}
    else:
        return {'marker': 'o', 'ms':6, 'mew':2, 'mec': color ,'mfc': 'white'}

def setup_twinax(color1='black', color2='black', ax=None):
    """
    Sets up a double axes plot (two y-axes, one x-axis). It also colors the axes according
    to color1 (left) and color2 (right). Returns the left and right axes.
    :param color1: String denoting the color of the left axis
    :param color2: String denoting the color of the right axis
    :return: 2 axis handles
    """
    if ax is None:
        ax = plt.gca()
    ax2 = ax.twinx()

    ax2.tick_params(axis='y', colors=color2)
    ax2.yaxis.label.set_color(color2)
    ax.tick_params(axis='y', colors=color1)
    ax.yaxis.label.set_color(color1)

    return ax, ax2

def plot_spectrum(y, t, ret=True, do_plot=True, freqlim='auto', ylim='auto', logscale=True, linear=True, type=None,
                  verbose=True, do_phase=False):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t). t should have evenly spaced entries, i.e. constant dt.
    Returns the spectrum.

    Example usage:

    t = arange(0,1,0.01)  # time vector
    y = sin(2*pi*t)
    plot_spectrum(y,t)

    Should produce a peak at 1 Hz.

    :param y: Amplitude in V.
    :param t: Time in seconds.
    :param ret: True/False. Default is True. Set to false if returning spectral data is not desired.
    :param do_plot: True/False. Plot the spectrum.
    :param freqlim: Tuple. Limits of the frequency axis.
    :param ylim: Tuple. Limits of y-axis.
    :param logscale: True/False for logarithmic y axis.
    :param linear: True/False for plotting in dB. When linear=False, automatically turns logscale to False
                   (negative dB cannot be converted to logscale)
    :param type: 'psd' or 'asd'. Default is 'asd'
    :param verbose: True/False, Prints the maximum contribution in the spectrum.
    :param do_phase: True/False. Plots a second figure with the Phase spectral density.
    :return: Frequency, ASD (complex) for type='asd', Frequency, PSD (real) for type='psd'
    """
    
    if len(t) < 2: 
        raise ValueError("Length of t is smaler than 2, cannot compute FFT.")
    else:
        dt = t[1]-t[0]
        n = len(y) # length of the signal
        T = n*dt
        frq = np.arange(n)/float(T) # two sides frequency range
        frq = frq[range(int(n/2))] # one side frequency range

        Y = np.fft.fft(y)/float(n) # fft computing and normalization
        Y = Y[range(int(n/2))] # maps the negative frequencies on the positive ones. Only works for real input signals!

        if do_plot:
            plt.figure(figsize = (6.,4.))
            ax1 = plt.subplot(111)
            configure_axes(13)
            if linear:
                if type == 'psd':
                    ax1.plot(frq, np.abs(Y)**2, 'r')
                    ax1.set_ylabel(r'Power spectral density (V$^2$)')
                else:
                    ax1.plot(frq, np.abs(Y), 'r') # plotting the spectrum
                    ax1.set_ylabel(r'Amplitude spectral density (V)')
                scientific_axes = 'both'
            else:
                if type == 'psd':
                    ax1.plot(frq, 20*np.log10(np.abs(Y)), 'r')
                    ax1.set_ylabel(r'Power spectral density (V$^2$)')
                else:
                    ax1.plot(frq, 10*np.log10(np.abs(Y)), 'r')
                    ax1.set_ylabel(r'Amplitude spectral density (V)')

                logscale = False
                scientific_axes = 'x'

            ax1.ticklabel_format(style='sci', axis=scientific_axes, scilimits=(0,0))

            if do_phase:
                plt.figure(figsize=(6.,4.))
                ax2 = plt.subplot(111)
                ax2.plot(frq, get_phase(Y), 'lightgreen')
                ax2.set_ylabel('Phase density (rad)')
                ax2.set_xlabel('$f$ (Hz)')

            ax1.set_xlabel('$f$ (Hz)')

            if logscale:
                ax1.set_yscale('log')

            if freqlim != 'auto':
                try:
                    ax1.set_xlim(freqlim)
                    if do_phase:
                        ax2.set_xlim(freqlim)
                except:
                    print("Not a valid xlim, please specify as [min, max]")

            if ylim != 'auto':
                try:
                    ax1.set_ylim(ylim)
                    if do_phase:
                        ax2.set_ylim(ylim)
                except:
                    print("Not a valid ylim, please specify as [min, max]")


        if verbose:
            print("Maximum contribution to signal is %.2e for a frequency of %.2e Hz"\
                    %(np.max(np.abs(Y)), frq[np.where(np.max(np.abs(Y)))[0]]))
    
        if ret:
            if type!='psd':
                return frq,Y
            else:
                return frq, np.abs(Y)**2

def filter_in_time(time_axis, voltage_axis, filter_type='RC_double_low_pass', do_plot=False, **kwargs):
    """
    Calculates the response of a voltage wave form sent through a filter. The filter type can be specified by
    filter_type.
    :param time_axis: numpy array that defines the x-axis
    :param voltage_axis: wave form must have the same length as time_axis
    :param filter_type: one of 'RC_low_pass', 'RC_double_low_pass', 'RC_high_pass'
    :param do_plot: plot the resulting filtered waveform
    :param kwargs: Arguments passed on to the filter function such as R, C (RC_low_pass/RC_high_pass) or
    R1, C1, R2, C2 (RC_double_low_pass)
    :return:
    """

    if filter_type == 'RC_low_pass':
        def filter_function(f, **kwargs):
            omega = 2 * np.pi * f
            R = kwargs['R']
            C = kwargs['C']
            return 1 / (1 + 1j * omega * R * C)

    elif filter_type == 'RC_double_low_pass':
        def filter_function(f, **kwargs):
            omega = 2 * np.pi * f
            R1 = kwargs['R1']
            R2 = kwargs['R2']
            C1 = kwargs['C1']
            C2 = kwargs['C2']
            return 1 / (1 + 1j * omega * (R1 * C1 + R2 * C2) + omega **2 * (R1 * C1 * R2 * C2))

    elif filter_type == 'RC_high_pass':
        def filter_function(f, **kwargs):
            omega = 2 * np.pi * f
            R = kwargs['R']
            C = kwargs['C']
            return 1 / (1 + 1 / (1j * omega * R * C))

    # Data must be even in length, otherwise it'll lead to problems.
    if (len(time_axis) % 2):
        time_axis = time_axis[:-1]
        voltage_axis = voltage_axis[:-1]

    # We use plot_spectrum to calculate the frequency points
    f, Y = plot_spectrum(voltage_axis, time_axis, verbose=False, ret=True, do_plot=False)\

    Vfft = np.fft.fft(voltage_axis) # FFT of the input signal
    Ffft = np.zeros(len(Vfft), dtype=np.complex128) # FFT of the RC filter must have the same form as the output of np.fft.fft
    Ffft[:len(Vfft) // 2] = filter_function(f, **kwargs)
    Ffft[len(Vfft) // 2 + 1:] = filter_function(f, **kwargs)[1:][::-1]

    if do_plot:
        plt.plot(time_axis, np.fft.ifft(Vfft), label='untouched')
        plt.plot(time_axis, np.fft.ifft(Vfft * Ffft), label='filtered')

    return time_axis, np.fft.ifft(Vfft * Ffft)


def split_power(power_in, conversion_loss):
    """
    Calculates the power at the output of a power splitter.
    :param power_in: power at the input in dBm
    :param conversion_loss:
    :return:
    """
    power_in_dB = power_in-conversion_loss
    inW = 0.5*10**(power_in_dB/10.)
    power_out_dB = 10*np.log10(inW)
    return power_out_dB

def Qext(L, C, Cin, Cout):
    """
    Calculates the external Q value of the resonator from the values 
    of the coupling capacitors, and the specifications of the resonator. 
    """
    w0 = 1/np.sqrt(L*C)
    Z = np.sqrt(L/C)
    Z0 = 50
    print("Resonance frequency = f0 = %.3e Hz"%(w0/(2*np.pi)))
    print("Impedance = Z = %.3e Ohms"%Z)
    return 1/(Z0*w0**3*L*(Cin**2+Cout**2))

def CfromQ(L, C, Qext):
    """
    Calculates the coupling capacitance for a desired external Q value. 
    Input parameters are L and C values of the resonator. 
    NOTE: This assumes that both coupling capacitors (Cin and Cout) are the same. 
    """
    w0 = 1/np.sqrt(L*C)
    Z = np.sqrt(L/C)
    Z0 = 50
    print("Resonance frequency = f0 = %.3e Hz"%(w0/(2*np.pi)))
    print("Impedance = Z = %.3e Ohms"%Z)
    return 1/np.sqrt(2*Qext*Z0*w0**3*L)

def q_finder(magnitudes, fpoints, debug=False, start_idx=None):
    """
    Method to find the Q factor of a frequency vs magnitude (dB) trace.
    :param magnitudes: Magnitude of the transmission peak in dB
    :param fpoints: Frequency in Hz
    :param debug: Prints the magnitude at the current time step
    :param start_idx: Point at which the searching for the -3dB points should begin. Default is the maximum of the curve.
    :return: Q
    """
    if start_idx is None: 
        start_idx = np.where(magnitudes == np.max(magnitudes))[0][0]
    else: 
        pass 

    max_mag = magnitudes[start_idx]
    if debug: print("Maximum magnitude is %.2f"%max_mag)
    f0 = fpoints[start_idx]
    walking_magnitude = max_mag
    k=1

    while abs(walking_magnitude-max_mag) < 3:
        walking_magnitude = magnitudes[start_idx+k]
        if debug: print(walking_magnitude)
        k+=1

    threedbpointright = fpoints[start_idx+k]
    if debug: print(threedbpointright)
    k=1

    while abs(walking_magnitude-max_mag) < 3:
        walking_magnitude = magnitudes[start_idx-k]
        k+=1
    
    threedbpointleft = fpoints[start_idx-k]
    if debug: print(threedbpointleft)

    df = threedbpointright-threedbpointleft
    return f0/df

def get_thermal_photons(f, T):
    """
    Returns the number of thermal photons at frequency f and temperature T.
    :param f: Frequency of the photons
    :param T: Temperature of the bath
    :return:
    """
    kB = 1.38E-23
    h = 6.63E-34
    return (kB*T)/(h*f)

def get_noof_photons_in_cavity(P, f0, Q):
    """
    Returns the number of photons in the cavity on resonance
    :param P: Input power in the drive in dBm
    :param f0: Resonant frequency of the cavity
    :param Q: Q of the cavity
    :return:
    """
    hbar = 1.055E-34
    w0 = 2*np.pi*f0
    P_W = 10**((P-30)/10.)
    kappa = w0/Q
    return P_W/(hbar*w0*kappa)

def get_noof_photons_in_input(P, f):
    """
    Returns the number of photons in the input drive for a given power P and frequency f.
    :param P: Input power in the drive in dBm
    :param f: Frequency of the input photons
    :return:
    """
    hbar = 1.055E-34
    w0 = 2*np.pi*f
    P_W = 10**((P-30)/10.)
    return P_W/(hbar*w0**2/(2*np.pi))

def pad_zeros(f, Y, until='auto', verbose=False):
    """
    Fill an array with zeros up to a specific index "until". Until may be "auto" which means the program will
    add zeros until the size of the new array is a power of two.
    :param f: Frequency domain data (1D array)
    :param Y: Amplitude data (1D or 2D array)
    :param until: 'auto' or an integer
    :param verbose: True/False
    :return:
    """
    if len(f) != len(Y):
        print("Expected arrays of same length, got different size arrays.")
    else:
        if until == 'auto':
            # Pad zeros until the nearest power of 2:
            Ni = len(Y)
            Nf = int(2**(np.ceil(np.log2(Ni))))
        elif isinstance(until, int):
            Nf = until
        else:
            raise ValueError('Until may be one of two things: auto or an integer.')

        if len(Y.shape) > 1:
            # 2D arrays
            Ynew = np.zeros((Y.shape[0], Nf))
            Ynew[:,:Ni] = Y

        else:
            Ynew = np.zeros(Nf)
            fnew = np.zeros(Nf)
            df = np.diff(f)[0]

            Ynew[:Ni] = Y
            fnew[:Ni] = f
            fnew[Ni:Nf] = np.linspace(f[-1]+df, f[0]+Nf*df, Nf-Ni)

        if verbose:
            try:
                print("New shape of array is (%d x %d)" % (Ynew.shape[0], Ynew.shape[1]))
            except:
                print("New length of array is %d" % (Ynew.shape[0]))

        return fnew, Ynew

def get_psd(t, y, verbose=False, window=True):
    """
    Computes the periodogram Only for real signals.
    :param t: Time in seconds
    :param y: Amplitude in V
    :param verbose: True/False. This will print(the value of the largest contribution to the PSD
    :param window: This will multiply the FFT with a Hanning window to reduce influence from the finite measurement time.
                   The defult is True. The Hanning window's efficiency is largest for finite size time traces.
                   More on Hanning windows here: https://en.wikipedia.org/wiki/Window_function#Hann_.28Hanning.29_window
    :return: frequency, PSD
    """

    if len(t) < 2:
        raise ValueError("Length of t is smaler than 2, cannot compute FFT.")
    else:
        dt = t[1]-t[0]
        n = len(y) # length of the signal
        T = n*dt
        frq = np.arange(n)/float(T) # two sides frequency range
        frq = frq[range(int(n/2))] # one side frequency range

        if window:
            y *= np.hanning(n)

        Y = np.fft.fft(y)*np.sqrt(T)/float(n) # fft computing and normalization
        Y = Y[range(int(n/2))] # maps the negative frequencies on the positive ones. Only works for real input signals!

        if verbose:
            print("Maximum contribution to signal is %.2e for a frequency of %.2e Hz"\
                    %(np.max(np.abs(Y)), frq[np.where(np.max(np.abs(Y)))[0]]))

        return frq, np.abs(Y)**2

def get_circular_points(radius, npts, theta_offset=0, do_plot=True):
    """
    Plots npts points in a circular fashion and prints their coordinates. Useful when you're machining a part.
    :param radius: Radius of the pattern
    :param npts: Number of points
    :param theta_offset: Additional rotation may be needed of the points.
    :param do_plot: True/False. Creates a plot of the points.
    :return: None
    """

    plt.figure(figsize=(5.,4.5))
    configure_axes(13)

    x, y = list(), list()
    for n in range(npts):
        xn = radius*np.cos((theta_offset + (n+1) * 360./npts) * np.pi/180.)
        yn = radius*np.sin((theta_offset + (n+1) * 360./npts) * np.pi/180.)

        x.append(xn)
        y.append(yn)

        plt.plot(xn, yn, 'or')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.xlim(-1.05*radius, 1.05*radius)
    plt.ylim(-1.05*radius, 1.05*radius)

    t = np.linspace(-180., 180., 100)
    xcircle = radius*np.cos(t * np.pi/180.)
    ycircle = radius*np.sin(t * np.pi/180.)
    plt.plot(xcircle, ycircle, '--r')

    print(tabulate(zip(x, y), headers=["x", "y"], tablefmt="rst", floatfmt=".4f", numalign="center", stralign='center'))