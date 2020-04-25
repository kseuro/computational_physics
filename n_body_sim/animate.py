import time
import random
import numpy as np
from scipy.integrate import RK45 as ODE45
from scipy.integrate import solve_ivp as IVP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

plt.rcParams['axes.facecolor']='#000000'

N = 1000

def update_anim(i, ast_data, dots) :
    time = i
    ax.set_xlabel(f'X [AU]\nSteps:{i} years', color = '#FFFFFF')
    ax.set_ylabel(f'Y [AU]\nSteps:{i} years', color = '#FFFFFF')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0))
    for dot, data in zip(dots, ast_data) :
            dot.set_data((data[0][i], data[1][i]))
            dot.set_3d_properties(data[2][i])
    return dots

if __name__ == '__main__':
    ast_data = []

    # Read asteroid data
    for asteroid in range(0, N):
        data = np.loadtxt(f"./data/asteroids/asteroid{asteroid}.txt")
        ast_data.append([data[0], data[1], data[2]])

    # Convert lists to single numpy array
    ast_data = np.array(ast_data) # Shape: [N, 3]

    # Generate Sun Data
    sun_data = np.copy(ast_data)
    sun_data.fill(0)

    jup_data = []

    # Read jupiter data
    data = np.loadtxt(f"./data/jupiter/jupiter0.txt")
    jup_data.append([data[0], data[1], data[2]])

    # Convert lists to single numpy array
    jup_data = np.array(jup_data) # Shape: [N, 3]

    fig  = plt.figure()
    ax   = Axes3D(fig)

    # Plot asteroid data
    dots = [ax.plot(dat[0], dat[1], dat[2], 'o', markersize=2, color='blue', alpha = 0.75)[0] for i, dat in enumerate(ast_data)]

    # Plot Jupiter data
    jups = [ax.plot(dat[0], dat[1], dat[2], 'o', markersize=10, color='orange')[0] for i, dat in enumerate(jup_data)]

    # Plot Sun data
    suns = [ax.plot(dat[0], dat[1], dat[2], 'o', markersize=30, color='yellow', alpha = 0.50)[0] for i, dat in enumerate(sun_data)]

    # Put the solar system back together
    ast_data = np.concatenate((ast_data, sun_data, jup_data), axis=0)
    dots.extend(suns)
    dots.extend(jups)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)

    anim = animation.FuncAnimation(fig, update_anim, ast_data[0][0].size, fargs=(ast_data, dots), interval=1, blit=False)

    anim.save("restricted_{}-Body.mp4".format(N), writer=writer, savefig_kwargs={'transparent': True, 'facecolor': '#000000'})
