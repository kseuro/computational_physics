import time
import random
import numpy as np
from scipy.integrate import RK45 as ODE45
from scipy.integrate import solve_ivp as IVP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

plt.rcParams['axes.facecolor']='#000000'

random.seed(time.time())

# Constants
N = 1000
G = 6.67e-11 * ( (365*86400)**2 / (1.5e11)**3 )
M_S = 1.99e30
AU = 1.5e11
M_J = 1.898e27
R_J = 5.203
V_J = 13720 * 365 * 86400
M_Ast = (10e10, 10e20)
AUs = (2.1, 3.2)

# Sun
x0_S = 0 # [AU]
y0_S = 0
vx0_S = 0
vy0_S = 0 # [AU/Yr]

# State vector
s0 = [vx0_S, vy0_S, x0_S, y0_S, 0, V_J, R_J, 0]

# Integration time
t = 10000; t_span = (0, t); first_step = 1.0; max_step = np.inf;

# Integration function args
options = {'rtol':1e-8, 'atol':1e-12, 'max_step': np.inf}

# Derivative Function
def D_grav(t, s_vec, m_arr, G):
    '''
        Function that computes the derivatives of an N-Body Gravity problem.
        Will be passed to the ODE45 algorithm as an argument.
        Args:
            - t (float): starting time for the numerical integrator. Assumed to begin at one
                         and integrate forwards in time. Unless you're Doc Martin, of course.
            - s_vec (array): system intitial state vector
                - This array needs to have the form: [V_x0, V_y0, X_0, Y_0] repeated for each object.
                  The algorithm assumes that every four entries within the array correspond to a single
                  object, so the entries need to align accordingly.
                - Example: 0: vx0_S, 1: vy0_S, 2: x0_S, 3: y0_S
            - m_arr (array): Masses of the Sun, Jupiter, and a single asteroid.
                - The array needs to have the form: [M_S, M_J, M_a].
                  The algorithm assumes that the mass information is located at the corresponding
                  indices in the array.
        Returns:
            - dsdt (Numpy array): Array of derivative values for each of the n-asteroids in the simulation.
                                  The array has the form (N*4,):
                                  [dvx/dt_0, dvy/dt_0, dx/dt_0, dy/dt_0, ... , dvx/dt_N, dvy/dt_N, dx/dt_N, dy/dt_N]
    '''
    verbose = False

    # Set the range of the for-loops to the length of the mass array
    N = len(m_arr)
    assert N==3, "Mass array must be of length 3 only!"

    # Set the size of the final storage array to N*4 in order to accomodate
    # vx, vy, x, y data for each of the three objects
    M = N*4

    # Arrays to track the position computations
    # - x-velocity - # - y-velocity - # - x-coords - # - y-coords - #
    vx = np.zeros(N); vy = np.zeros(N); x = np.zeros(N); y = np.zeros(N)

    # Arrays to track the derivative computations
    # --- dvx/dt --- # ----- dvy/dt ----- # ----- dx/dt ----- # --- dy/dt --- #
    dvx = np.zeros(N); dvy = np.zeros(N); dx = np.zeros(N); dy = np.zeros(N)

    # Again, this loop assumes that every four entries in the state vector
    # correspond to a single object. This algorithm will fail if the state
    # vector is not constructed properly.
    idx = 0; place = 0
    for i in range(0, N):
        vx[i] = s_vec[place]
        vy[i] = s_vec[place+1]
        x[i]  = s_vec[place+2]
        y[i]  = s_vec[place+3]
        idx += 1
        place += 4

    # Find the total force acting on an asteroid from the Sun and Jupiter.
    for i in range(0, N):
        # Position derivates are given by the current velocity
        dx[i] = vx[i]
        dy[i] = vy[i]
        ## Set initial acceleration to zero and sum over the force contributions
        ## from all of the masses (Sun, Jupiter, Asteroid)
        a_x = 0; a_y = 0
        for k in range(0, N):
            if i != k:
                # Find the separation distance
                r_sep = (x[i] - x[k])**2 + (y[i] - y[k])**2
                # Compute the x-y components of the acceleration of the ith mass due to the kth mass
                a_x = a_x + ((G * m_arr[k] * (x[k] - x[i])) / r_sep**(3/2))
                a_y = a_y + ((G * m_arr[k] * (y[k] - y[i])) / r_sep**(3/2))
        # Store the computed acceleration values
        dvx[i] = a_x
        dvy[i] = a_y

    # Create a single vector from all of the computed values
    dsdt = np.zeros(M)

    idx = 0; place = 0
    for i in range(0, N):
        dsdt[place]   = dvx[i]
        dsdt[place+1] = dvy[i]
        dsdt[place+2] = dx[i]
        dsdt[place+3] = dy[i]
        idx += 1
        place += 4

    return dsdt

def random_float(low, high):
    '''
        Function that returns a random floating point number in the range [low, high] inclusive.
        Could use Numpy to do this, but it's more interesting to write my own function.
        Args:
            - low (float): lower bound of desired random float.
            - high (float): upper bound of desired random float.
        Returns:
            - random floating point number
    '''
    return random.random()*(high-low) + low

def make_asteroids(N, G, M_S, M_Ast, AUs):
    '''
        - This function generates randomized arrays corresponding to asteroid masses, their position components
          in cartesian coordinates [AU], and velocity components [AU/Yr]
        Args:
            - N (int): number of asteroids to generate
            - G (float): gravitational constant
            - M_S (float): mass of the sun used to generate random asteroid velocities
            - M_Ast (tuple): 2-tuple of floats representing the low and high end of the masses
                             of asteroids in the asteroid belt
            - AUs (tuple): 2-tuple of astronomical unit used to compute starting positions
        Returns:
            - state_vector (dict) containing:
                - a_masses (Numpy Array): array of N masses, each randomly distributed between 10^10 and 10^20 kg
                - a_x (Numpy Array): array of randomly distributed x-coordinates for N asteroids
                - a_y (Numpy Array): array of randomly distributed y-coordinates for N asteroids
                - v_x (Numpy Array): array of randomly distributed x-velocities for N asteroids
                - v_y (Numpy Array): array of randomly distributed y-velocities for N asteroids
    '''
    # Initialize the arrays
    # ----- Masses ----- ## -- x-coords -- ## -- y-coords -- ## -- x-velocity -- ## -- y-velocity -- #
    a_masses = np.zeros(N); a_x = np.zeros(N); a_y = np.zeros(N); v_x = np.zeros(N); v_y = np.zeros(N)
    # ---------------------------------------------------------------------------------------------- #

    print(f'Generating {N} Asteroid(s)')
    # Loop over N asteroids and randomly allocate coordinates and velocities
    for i in range(N):
        # Randomly assign an asteroid mass in the given range using a uniform distribution
        a_masses[i] = random_float(low = M_Ast[0], high = M_Ast[1])

        # Randomly assign a starting position in AU from the Sun using a uniform distribution
        r = random_float(low = AUs[0], high = AUs[1])

        # Generate a random starting angle between 0 and 2pi
        theta = 2*np.pi*random_float(low = 0.0, high = 1.0)

        # Determine the velocity of the asteroid using Kepler's Third Law
        # The period square is proportional to the cube of the semi-major axis, but for this
        # simulation we assume a circular orbit for simplicity.
        P = np.sqrt(r**3)

        # The velocity is the circumference of the circular orbit divided by the period
        v = (2*np.pi*r) / P

        # Use the velocity and a trigonometric relationship to find the starting
        # coordinates and starting velocity of the asteroid. We take the orbits
        # to be anti-clockwise, so the starting x-velocity is in the negative direction
        a_x[i] = r*np.cos(theta)
        a_y[i] = r*np.sin(theta)

        v_x[i] = -v*np.sin(theta)
        v_y[i] = v*np.cos(theta)

    return {'a_masses': a_masses, 'ax': a_x, 'ay': a_y, 'vx': v_x, 'vy': v_y}

if __name__ == '__main__':

    # Prepare lists for storing results
    r_init = []; r_final = []; x_coords = []; y_coords = []

    for i in range(0, N):
        # Periodically update progress to terminal
        if N >= 1000 and i % 1000 == 0:
            print("Asteroid: {}".format(i))
        elif N <= 500 and i % 100 == 0:
            print("Asteroid: {}".format(i))
        # Calculate the asteroid's initial distance from the Sun
        r_init.append(np.sqrt(asteroids['ax'][i]**2 + asteroids['ay'][i]**2 ))

        # Generate the initial state of the asteroid-Jupiter-Sun system
        # We are not interested in the graviational interactions of
        # the asteroids themselves, since this contribution is negligible
        # compared with the contributions from Jupiter and the Sun.
        y0 = s0 + [asteroids['vx'][i], asteroids['vy'][i], asteroids['ax'][i], asteroids['ay'][i]]

        # Include the arrays of masses corresponding to the three objects
        masses = [M_S, M_J, asteroids['a_masses'][i]]

        # Run the numerical integration
        ## The returned solution is a dictionary containing arrays
        ## t: - the time over which the solution is integrated
        ## y: - the solution of the IVP at each timestep
        solution = IVP(D_grav, t_span, y0, method='RK45', args=(masses, G), **options)

        # Compute the final position of the asteroid after integration
        r_final.append(np.sqrt(solution['y'][-1][-1]**2 + solution['y'][-1][-2]**2))

        # Get the x- and y-coordinates for the asteroids
        x_coords.append(solution['y'][-2])
        y_coords.append(solution['y'][-1])

    x_coords = np.asarray(x_coords)
    y_coords = np.asarray(y_coords)

    min_x = min([x_coords[k].shape[0] for k in range(N)])

    z_coords = np.zeros((min_x))
    for i in range(0, N):
        data = np.vstack((x_coords[i][0:min_x], y_coords[i][0:min_x], z_coords))
        np.savetxt("./data/asteroids/asteroid{}.txt".format(i), data)

    n = min_x
    x_jup = [np.cos(2*np.pi/n*x)*R_J for x in range(0, n+1)]
    y_jup = [np.sin(2*np.pi/n*x)*R_J for x in range(0, n+1)]
    data = np.vstack((x_jup[0:min_x], y_jup[0:min_x], z_coords))
    np.savetxt("./data/jupiter/jupiter{0}.txt", data)

    fig, ax = plt.subplots()
    plt.title("Initital Distribution of {} Asteroids\n between [{}, {}] AU".format(N, AUs[0], AUs[1] ))
    ax.hist(r_init)
    ax.set_xlabel('[AU]')
    ax.set_ylabel('Counts')
    plt.savefig("Initital_Distribution.png", dpi = 300)
    plt.close()

    fig, ax = plt.subplots()
    plt.title("Final Distribution of {} Asteroids\n between [{}, {}] AU".format(N, AUs[0], AUs[1] ))
    ax.hist(r_final)
    ax.set_xlabel('[AU]')
    ax.set_ylabel('Counts')
    plt.savefig("Final_Distribution.png", dpi = 300)
    plt.close()
