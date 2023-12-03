import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import interpolate
from time import time

from schrodinger import schrodinger

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

folder = 'C:/frames/' + name

res = np.array((1920, 1080))
res = res if record else res // 1.25

sim = schrodinger.Simulate(sim_size, collapse=do_collapse)
frames = fps * sec

if record:
    do_smoothing = True

plt.rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg/bin/ffmpeg.exe'
plt.rcParams['animation.writer'] = 'ffmpeg'

real_fig, real_ax = plt.subplots()
imag_fig, imag_ax = plt.subplots()

def cubic_interp1d(x0, x, y):
    f = interpolate.interp1d(x, y, kind='cubic')
    return f(x0)

def make_wavelines(wavedata, P=5000, axis=0, stride=4, smoothing=True):
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

    for x in range(N):
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
    """
    Generate a smoothed surface based on input surface data.
    
    Parameters:
        surf_data (ndarray): The input surface data as a 2D array.
        smoothing (int): The level of smoothing to apply to the surface (default is 2).
        
    Returns:
        ndarray: The smoothed surface data as a 2D array.
    """
    X = surf_data.shape[0]
    Y = surf_data.shape[1]
    x, y = np.arange(X), np.arange(Y)
  
    f = interpolate.RectBivariateSpline(x, y, surf_data, kx=3, ky=3)

    xnew = np.linspace(0, X, X*smoothing)
    ynew = np.linspace(0, Y, Y*smoothing)
   
    return f(xnew, ynew)

def update(frame):
    global index

    if index >= frames and record:
        real_fig.savefig(f'{folder}/real_{index}.png')
        imag_fig.savefig(f'{folder}/imag_{index}.png')
        return

    d = sim.simulate_frame(debug=0)

    zdata = np.abs(d) ** 2

    if do_smoothing:
        zdata = surf_smoothing(zdata, smoothing=1)

    zcol = plt.cm.viridis(zdata)
    zcol[:, :, 3] = zdata + .1
    zcol[:, :, 3] *= 0.75

    real_ax.clear()
    real_ax.imshow(zdata, cmap='viridis', extent=[0, sim_size, 0, sim_size], origin='lower', alpha=0.75)

    dreal, dimag = d.real, np.flipud(d.imag)
    rpoints, realcolors = make_wavelines(dreal, axis=0, smoothing=do_smoothing)
    ipoints, imagcolors = make_wavelines(dimag, axis=1, smoothing=do_smoothing)

    realcolors[:, 0:3:2] *= 0.7
    imagcolors[:, 0] *= 0.3

    real_ax.plot(rpoints[:, 0], rpoints[:, 1], color='red', alpha=0.33)
    real_ax.plot(ipoints[:, 0], ipoints[:, 1], color='blue', alpha=0.33)

    index += 1

ani = FuncAnimation(real_fig, update, frames=frames, interval=1000/fps, repeat=False)
plt.show()
