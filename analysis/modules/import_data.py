import numpy as np
from matplotlib import pyplot as plt
import re
from . import common, kfit


def load_dsp(df):
    """
    Loads a .dsp file from Maxwell and extracts elements, nodes and the solution at the nodes.
    For this code to work, the data must have been saved as a dsp file, with only a single plot in the Fields tab.
    :param df: File path of the data file
    :return: elements, node, element solution, bounding box
    """
    with open(df, 'r') as myfile:
        data = myfile.readlines()

    # The important data is stored on line numbers 91-93.
    # Line 91: Elements: Each element is composed of 6 nodes. Each sequence of 2,3,3,0,6 is followed by 6 points, which will
    # make up a single element. First 2 entries are diagnostic info.
    # Line 92: Node coordinates. One node coordinate has 3 entries: x, y, z
    # Line 93: Solution on each node. First 3 entries are diagnostic info.

    line_nr = [91, 92, 93]
    elements = np.array(re.findall(r"\((.*?)\)", data[line_nr[0]-1])[0].split(', '), dtype=int)
    nodes = np.array(re.findall(r"\((.*?)\)", data[line_nr[1]-1])[0].split(', '), dtype=float)
    elem_solution = np.array(re.findall(r"\((.*?)\)", data[line_nr[2]-1])[0].split(', '), dtype=float)

    nodes = nodes.reshape((int(nodes.shape[0]/3), 3))

    line_nr = 90
    bounding_box = np.array(re.findall(r"\((.*?)\)", data[line_nr-1])[0].split(', '), dtype=float)

    return elements, nodes, elem_solution[3:], bounding_box

def select_domain(X, Y, Esquared, xdomain=None, ydomain=None):
    """
    Selects a specific area determined by xdomain and ydomain in X, Y and Esquared. X, Y and Esquared may be
    obtained from the function load_maxwell_data. To retrieve a meshgrid from the returned 1D arrays x_cut and y_cut,
    use Xcut, Ycut = meshgrid(x_cut, y_cut)
    :param X: a 2D array with X coordinates
    :param Y: a 2D array with Y coordinates
    :param Esquared: Electric field squared. Needs to be the same shape as X and Y
    :param xdomain: Tuple specifying the minimum and maximum of the x domain
    :param ydomain: Tuple specifying the minimum and the maximum of the y domain
    :return:
    """

    if xdomain is None:
        xmin = np.min(X[:,0])
        xmax = np.max(X[:,0])
        if ydomain is None:
            ymin = np.min(Y[0,:])
            ymax = np.max(Y[0,:])
    elif ydomain is None:
        ymin = np.min(Y[0,:])
        ymax = np.max(Y[0,:])
    else:
        xmin, xmax = xdomain
        ymin, ymax = ydomain

    if np.shape(X) == np.shape(Y) == np.shape(Esquared):
        if len(np.shape(X)) > 1 and len(np.shape(Y)) > 1:
            x = X[:,0]
            y = Y[0,:]
        else:
            print("Shapes of X, Y and Esquared are not consistent:\nShape X: %s \nShape Y: %s\nShape Esquared: %s" % (
            np.shape(X), np.shape(Y), np.shape(Esquared)))
            return

        x_cut = x[np.logical_and(x>=xmin, x<=xmax)]
        y_cut = y[np.logical_and(y>=ymin, y<=ymax)]

        xidx = np.where(np.logical_and(x>=xmin, x<=xmax))[0]
        yidx = np.where(np.logical_and(y>=ymin, y<=ymax))[0]

        Esquared_cut = np.transpose(Esquared[xidx[0]:xidx[-1]+1, yidx[0]:yidx[-1]+1])

        return x_cut, y_cut, Esquared_cut
    else:
        print(
            "Shapes of X, Y and Esquared are not consistent:\nShape X: %d x %d\nShape Y: %d x %d\nShape Esquared: %d "
            "x %d "
            % (np.shape(X)[0], np.shape(X)[1], np.shape(Y)[0], np.shape(Y)[1], np.shape(Esquared)[0],
               np.shape(Esquared)[1]))


def load_maxwell_data(df, do_plot=True, do_log=True, xlim=None, ylim=None, clim=None,
                       figsize=(6.,12.), plot_axes='xy', cmap=plt.cm.Spectral):
    """
    :param df: Path of the Maxwell data file (fld)
    :param do_plot: Use pcolormesh to plot the 3D data
    :param do_log: Plot the log10 of the array. Note that clim has to be adjusted accordingly
    :param xlim: Dafaults to None. May be any tuple.
    :param ylim: Defaults to None, May be any tuple.
    :param clim: Defaults to None, May be any tuple.
    :param figsize: Tuple of two floats, indicating the figure size for the plot (only if do_plot=True)
    :param plot_axes: May be any of the following: 'xy' (Default), 'xz' or 'yz'
    :return:
    """

    data = np.loadtxt(df, skiprows=2)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    magE = data[:,3]

    # Determine the shape of the array:
    if 'x' in plot_axes:
        for idx, X in enumerate(x):
            if X != x[0]:
                ysize=idx
                xsize=np.shape(magE)[0]/ysize
                break
    else:
        for idx, Y in enumerate(y):
            if Y != y[0]:
                ysize=idx
                xsize=np.shape(magE)[0]/ysize
                break

    # Cast the voltage data in an array:
    if plot_axes == 'xy':
        X = x.reshape((xsize, ysize))
        Y = y.reshape((xsize, ysize))
    if plot_axes == 'xz':
        X = x.reshape((xsize, ysize))
        Y = z.reshape((xsize, ysize))
    if plot_axes == 'yz':
        X = y.reshape((xsize, ysize))
        Y = z.reshape((xsize, ysize))

    E = magE.reshape((xsize, ysize))

    if do_plot:
        plt.figure(figsize=figsize)
        common.configure_axes(15)
        if do_log:
            plt.pcolormesh(X*1E6, Y*1E6, np.log10(E), cmap=cmap)
        else:
            plt.pcolormesh(X*1E6, Y*1E6, E, cmap=cmap)

        plt.colorbar()

        if clim is not None:
            plt.clim(clim)
        if xlim is None:
            plt.xlim([np.min(x)*1E6, np.max(x)*1E6]);
        else:
            plt.xlim(xlim)
        if ylim is None:
            plt.ylim([np.min(y)*1E6, np.max(y)*1E6]);
        else:
            plt.ylim(ylim)
        plt.xlabel('x ($\mu\mathrm{m}$)')
        plt.ylabel('y ($\mu\mathrm{m}$)')

    return X, Y, E