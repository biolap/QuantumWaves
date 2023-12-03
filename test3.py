import os
import sys

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication

import numpy as np
from pyqtgraph.Qt import QtCore
from pyqtgraph import Vector
from pathlib import Path
from math import sqrt
from matplotlib import pyplot as plt
from numba import njit, prange
from scipy import interpolate
from time import time


from schrodinger import schrodinger

do_parallel = False
do_collapse = False
sim_size = 125

index = 0
sec = 30
fps = 30
collapse_interval = (fps * sec) / 2
record = False
do_smoothing = True
layer = ''

if len(sys.argv) >= 2:
    layer = sys.argv[1]

name = 'decay_movement9_follow'
name = name + ('_' + layer) if layer != '' else name

folder = Path('C:/frames') / name

res = np.array((1920, 1080))
res = res if not record else res // 1.25

sim = schrodinger.Simulate(sim_size, collapse=do_collapse)
frames = fps * sec

pg.setConfigOptions(antialias=True)

app = QApplication([])
w = gl.GLViewWidget()
w.resize(int(res[0]), int(res[1]))
w.setFixedSize(int(res[0]), int(res[1]))

w.show()
w.setWindowTitle(
    f'{os.path.basename(__file__)} ... {sim_size}x{sim_size} ... {f"RENDERING {name}" if record else "PREVIEW"}'
)

w.setCameraPosition(distance=80)
w.resize(int(res[0]), int(res[1]))
w.setFixedSize(int(res[0]), int(res[1]))

screen_width = app.desktop().screenGeometry().width()
x = int(screen_width / 2 - res[0] / 2)
screen_height = app.desktop().screenGeometry().height()
y = int(screen_height / 2 - res[1] / 2)

w.move(x, y)

timer = QtCore.QTimer()

@njit(cache=True)
def cubic_interp1d(x0, x, y):
    """
    Perform cubic interpolation at a given point x0 using the x and y arrays.

    Parameters:
        x0 (float): The x-coordinate where interpolation is performed.
        x (ndarray): The array of x-coordinates.
        y (ndarray): The array of y-coordinates.

    Returns:
        float: The interpolated value at the point x0.
    """
    # Sort x and y arrays if x is not in ascending order
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)
    xdiff = np.diff(x)
    ydiff = np.diff(y)
    Li = np.empty(size)
    Li_1 = np.empty(size-1)
    z = np.empty(size)

    # Calculate Li, Li_1, and z arrays
    Li[0] = sqrt(2*xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0
    z[0] = B0 / Li[0]
    for i in range(1, size-1, 1):
        Li_1[i] = xdiff[i-1] / Li[i-1]
        Li[i] = sqrt(2*(xdiff[i-1]+xdiff[i]) - Li_1[i-1] * Li_1[i-1])
        Bi = 6*(ydiff[i]/xdiff[i] - ydiff[i-1]/xdiff[i-1])
        z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]
    i = size - 1
    Li_1[i-1] = xdiff[-1] / Li[i-1]
    Li[i] = sqrt(2*xdiff[-1] - Li_1[i-1] * Li_1[i-1])
    Bi = 0.0
    z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]
    i = size-1
    z[i] = z[i] / Li[i]
    for i in range(size-2, -1, -1):
        z[i] = (z[i] - Li_1[i-1]*z[i+1])/Li[i]

    # Perform interpolation
    index = np.searchsorted(x, x0)
    xi1, xi0 = x[index], x[index-1]
    yi1, yi0 = y[index], y[index-1]
    zi1, zi0 = z[index], z[index-1]
    hi1 = xi1 - xi0
    f0 = zi0/(6*hi1)*(xi1-x0)**3 + \
         zi1/(6*hi1)*(x0-xi0)**3 + \
         (yi1/hi1 - zi1*hi1/6)*(x0-xi0) + \
         (yi0/hi1 - zi0*hi1/6)*(xi1-x0)

    return f0

@njit(cache=True)
def make_gridlines(X, Y, axis=0, stride=4, extend=3):
    """
    Generates a grid of lines for a given X and Y coordinate range.

    Args:
        X (int): The X coordinate range.
        Y (int): The Y coordinate range.
        axis (int, optional): The axis along which to generate the grid lines. Defaults to 0.
        stride (int, optional): The spacing between each line. Defaults to 4.
        extend (int, optional): The number of lines to extend beyond the coordinate range. Defaults to 3.

    Returns:
        tuple: A tuple containing two numpy arrays: `points` and `colors`. `points` is a 2D array of shape (M, 3) where M is the total number of points in the grid. `colors` is a 2D array of shape (M, 4) where M is the total number of points in the grid.
    """
    if X != Y:
        raise Exception('X must equal Y')

    P = 4
    N = (2 * X) // stride
    M = N * P

    points = np.zeros((M, 3))
    colors = np.ones((M, 4))

    for n in prange(N):
        i0 = n * P
        i1 = n * P + n * P - 1

        a = int(axis)
        b = int(not axis)

        if n < N // 2:
            points[i0 : (i1 + 1), a] = 0 - n * stride
            fading = (1 - n / (N // 2))
        else:
            points[i0 : (i1 + 1), a] = 0 + n * stride
            fading = (1 - (n // 2) / (N // 2))

        points[i0, b] = -Y
        points[i0 + 1, b] = 0
        points[i0 + 2, b] = Y
        points[i0 + 3, b] = Y * 2

        colors[i0, 3] = 0
        colors[i0 + 1, 3] = 0.25 * fading
        colors[i0 + 2, 3] = 0.25 * fading
        colors[i0 + 3, 3] = 0

        if n == 0 or n == (N - 1):
            colors[i0 + 1, 3] = 0
            colors[i0 + 2, 3] = 0

    return points, colors

@njit(cache=True, parallel=do_parallel)
def make_wavelines(wavedata, P=5000, axis=0, stride=4, smoothing=True):
    """
    Generate a set of wave lines based on the given wavedata.

    Parameters:
    - wavedata: numpy array
        The input wavedata.
    - P: int, optional (default=5000)
        The number of points in each wave line.
    - axis: int, optional (default=0)
        The axis along which the wave lines are generated.
    - stride: int, optional (default=4)
        The stride between each wave line.
    - smoothing: bool, optional (default=True)
        Whether to apply smoothing to the wave lines.

    Returns:
    - points: numpy array
        The generated points of the wave lines.
    - colors: numpy array
        The colors of the wave lines.
    """
    X = wavedata.shape[0]
    Y = wavedata.shape[1]
    N = X // stride

    if not smoothing:
        P = Y

    M = N * (P + 2)
    points = np.zeros((M, 3))
    colors = np.ones((M, 4))

    y0 = np.arange(0, X)

    if smoothing:
        y = np.linspace(0, X - 1, P)
    else:
        y = np.linspace(0, X - 1, X)

    for x in prange(N):
        w = x * stride

        i0 = x * (P + 2)
        i1 = x * (P + 2) + (P + 2) - 1

        points[i0: (i1 + 1), 0] = w
        points[i0, 1] = -Y
        points[i0, 2] = 0
        colors[i0, 3] = 0

        if axis == 0:
            z0 = wavedata[w, :]
        else:
            z0 = wavedata[:, w]

        if smoothing:
            z = cubic_interp1d(y, y0, z0)
        else:
            z = z0

        points[i0 + 1: i1, 1] = y
        points[i0 + 1: i1, 2] = z
        colors[i0 + 1: i1, 3] = .33

        points[i1, 1] = 2 * Y
        points[i1, 2] = 0
        colors[i1, 3] = 0

    return points, colors

def surf_smoothing(surf_data, smoothing=2):
    X = surf_data.shape[0]
    Y = surf_data.shape[1]
    x, y = np.arange(X), np.arange(Y)
  
    f = interpolate.RectBivariateSpline(x, y, surf_data, kx=3, ky=3)

    xnew = np.linspace(0, X, X*smoothing)
    ynew = np.linspace(0, Y, Y*smoothing)
   
    return f(xnew, ynew)

def update():
    """
    Updates the state of the animation.

    This function updates the animation by performing the following steps:
    - If the index is greater than or equal to the number of frames and recording is enabled, the application is closed.
    - If recording is enabled, the estimated time of arrival (ETA) is calculated based on the time it took to process the previous frame and the number of frames remaining.
    - The current index and ETA are printed.
    - The current frame is saved as an image file if recording is enabled.
    - If recording is not enabled, the current index, the time it took to process the frame, and the elapsed time since the last frame are printed.
    - The simulation is run to generate the next frame.
    - The intensity of the simulation data is calculated.
    - If smoothing is enabled, the intensity data is smoothed.
    - The intensity data is interpolated to generate color data.
    - The color data is adjusted to set transparency and bias.
    - The simulation data is split into real and imaginary components.
    - Lines are generated based on the real and imaginary components.
    - The color of the lines is adjusted based on bias.
    - The animation is updated based on the simulation data and line data.
    - The camera position is adjusted.
    - If the animation has a "follow" mode, the camera follows the intensity data.
    - The index is incremented.
    - If the simulation has "collapse" mode enabled and the current index is a multiple of the collapse interval, the wavefunction collapses.

    Returns:
        None
    """
    global surf, index, folder, last_time, start_time, record, fps, timer

    surf_smooth = 1  # или любое другое значение по умолчанию

    if sim_size > 350:
        surf_smooth = 8 if do_smoothing else 1

    zscale = 3

    if index >= frames and record:
        app.quit()

    if record:
        ETA = (time() - last_time) * (frames - index)
        ETA = divmod(int(ETA), 60)
        last_time = time()

        print(index, 'ETA', ETA)

        w.grabFrameBuffer().save(str(folder / f'{name}_{index}.png'))
    else:
        elapsed_time = time() - last_time
        print(index, 'TIME', time() - last_time, 's', 'ELAPSED', index / fps, 's')
        start_time = time()

    time()

    start_sim_time = time()
    d = sim.simulate_frame(debug=0)
    sim_time = time() - start_sim_time

    zdata = np.abs(d) ** 2

    if do_smoothing:
        zdata = surf_smoothing(zdata, smoothing=surf_smooth)

    time()

    def interpolate(x):
        """
        Interpolates the given value `x` using numpy's `interp` function.

        Parameters:
            x (float): The value to be interpolated.

        Returns:
            float: The interpolated value.
        """
        return np.interp(x, [0, 4], [0, 1])

    zcol = cmap(interpolate(zdata))
    zcol[:, :, 3] = zdata + .1
    zcol[:, :, 3] *= 0.75

    time()

    dreal, dimag = d.real, np.flipud(d.imag)
    rpoints, realcolors = make_wavelines(dreal, axis=0, smoothing=do_smoothing)
    ipoints, imagcolors = make_wavelines(dimag, axis=1, smoothing=do_smoothing)

    realcolors[:, 0:3:2] *= rcol_bias
    imagcolors[:, 0] *= icol_bias

    time()

    if do_collapse:
        surf.setData(z=zdata / zdata.max(), colors=zcol)
        surf.setData(z=3 * zdata / zdata.max(), colors=zcol)
    else:
        surf.setData(z=zdata, colors=zcol)

    real.setData(pos=rpoints, color=realcolors)
    imag.setData(pos=ipoints, color=imagcolors)

    time()

    dazim = -.09
    delev = +.005
    ddist = 0

    if 'follow' in name:
        follow(zdata)

    w.orbit(dazim, delev)
    w.setCameraPosition(distance=w.opts['distance'] + ddist)

    time()

    index += 1

    if sim.collapse and index % collapse_interval == 0 and index > 0 and index < collapse_interval * 2:
        sim.dual_collapse_wavefunction()

def follow(data, lookback=10, r=30):
    """
    Adjusts the camera position to follow the intensity data.

    This function adjusts the camera position based on the intensity data to keep the interesting part of the simulation
    in view.

    Parameters:
        data (ndarray): The intensity data.
        lookback (int): The number of previous frames to consider when determining the region of interest.
        r (float): The radius of the region of interest.

    Returns:
        None
    """
    global w

    data = np.mean(data[-lookback:], axis=0)
    y, x = np.indices(data.shape)

    center_y = int(np.sum(y * data) / np.sum(data))
    center_x = int(np.sum(x * data) / np.sum(data))

    w.setCameraPosition(center=(center_x, center_y, 0), distance=r)


