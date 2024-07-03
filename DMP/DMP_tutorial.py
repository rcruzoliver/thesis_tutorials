import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline


def create_demonstration_trajectory(t_eval):

    A_start = 0.10  # Starting position
    A_end = 0.00  # Ending position
    
    # Position and derivative arrays
    times = np.array([0.00, 0.50, 1.50, 2.00])
    positions = np.array([A_start, A_start, A_end, A_end])
    velocities = np.array([0.0, 0.0, 0.0, 0.0])  # Velocity at start and end
    
    # Cubic Hermite Spline interpolation
    spline = CubicHermiteSpline(times, positions, velocities)
    spline_dot = spline.derivative(1)
    spline_dot_dot = spline.derivative(2)
    
    # Evaluate the trajectories
    y = spline(t_eval)
    y_dot = spline_dot(t_eval)
    y_dot_dot = spline_dot_dot(t_eval)
    
    return y, y_dot, y_dot_dot

# Define Radial Basis Function (RBF)
def rbf(x, center, width):
    return np.exp(-0.5 * ((x - center) / width)**2)

# Compute the forcing term based on RBFs
def rbf_forcing_temporal(t, N, w, centers, widths, x_func, g, y0):
    num = 0
    den = 0
    x = x_func(t)
    for i in range(N):
        num += w[i] * rbf(x, centers[i], widths[i])
        den += rbf(x, centers[i], widths[i])
    f = (num/den) * x * (g-y0)
    return f.item()

# Define canonical system dynamics with tau
def canonical_system(t, x, alpha_x, tau):
    dxdt = -alpha_x * x / tau
    return dxdt

# Define transformation system dynamics with RBF-based forcing term and tau
def transformation_system(t, y, alpha_y, beta_y, g, f_func, tau):
    y1, y2 = y
    dy1dt = y2 / tau
    dy2dt = (alpha_y * (beta_y * (g - y1) - y2) + f_func(t)) / tau
    return [dy1dt, dy2dt]

def compute_f_target(y, y_dot, y_dot_dot, tau, alpha, beta, g):
    return tau**2 * y_dot_dot - alpha*(beta*(g-y)-tau*y_dot) 
    
def compute_ji(x, g, y0):
    return x*(g-y0)

def compute_gamma(x, center, width):
    P = len(x)  
    gamma = np.zeros((P, P))  
    for j in range(P):
        gamma[j, j] = rbf(x[j].item(), center, width)
    return gamma
    
def compute_weights(N, g, y0, y, y_dot, y_dot_dot, tau, alpha, beta, x):
    w = np.zeros(N)
    s = compute_ji(x, g, y0).reshape(-1, 1)
    f_target = compute_f_target(y, y_dot, y_dot_dot, tau, alpha, beta, g).reshape(-1, 1)
    for i in range(N):
        gamma = compute_gamma(x, centers[i], widths[i])
        temp1 = (s.T @ gamma @ f_target)
        temp2 = (s.T @ gamma @ s)
        w[i] = temp1.item()/temp2.item()
    return w, f_target

def y_dot_dmp(t, ts_func):
    return ts_func(t)[1]

def y_dot_dot_dmp(t, ts_func, dt=1e-5):
    return (y_dot_dmp(t + dt, ts_func) - y_dot_dmp(t - dt, ts_func)) / (2 * dt)

def x_func_mod(t, x_func):
    t1 = 0.65
    t2 = 1.20

    if t<t1:
        x_mod = x_func(t)
    elif t<t2:
        x_mod = x_func(t1)
    else:
        x_mod = x_func(t-(t2-t1))

    return x_mod

# Parameters for RBF forcing function
t_span = (0, 2)  # Time span for simulation 
t_eval = np.linspace(0, 2, 500)  # Points to evaluate solution 

# Generate demonstration trajectory
y_demo, y_dot_demo, y_dot_dot_demo = create_demonstration_trajectory(t_eval)

N = 50  # Number of RBFs
centers = np.linspace(0, 1, N)  # Centers of RBFs (equally spaced) # in X
widths = np.ones(N) * 0.1  # Widths of RBFs (equal for simplicity)

# Solve the DMP system with RBF-based forcing term and tau
init = [0.10, 0.0]  # Initial position and velocity
g = 0.0  # Goal position
alpha_y = 25.0  # Transformation system parameter (damping factor)
beta_y = alpha_y / 4.0  # Transformation system parameter (stiffness factor)
tau = 1.0  # Time scaling factor

alpha_x = 3.0

# Solve the canonical system using solve_ivp
canonical_system_sol = solve_ivp(lambda t, x: canonical_system(t, x, alpha_x, tau), t_span, [1.0], t_eval=t_eval, dense_output=True)
x_func = canonical_system_sol.sol

# Extract the solution [for plotting]
x_traj = np.array([x_func(t) for t in t_eval])

# weights = np.random.rand(N) * 50  # Weights of RBFs (scaled up for demonstration)
weights, f_target = compute_weights(N, g, init[0], y_demo, y_dot_demo, y_dot_dot_demo, tau, alpha_y, beta_y, x_traj)

# Solve the DMP system using solve_ivp
transformation_system_sol = solve_ivp(lambda t, y: transformation_system(t, y, alpha_y, beta_y, g, lambda t: rbf_forcing_temporal(t, N, weights, centers, widths, x_func, g, init[0]), tau), t_span, init, t_eval=t_eval, dense_output=True)
transformation_system_func = transformation_system_sol.sol
y_dot_dot_traj = np.array([y_dot_dot_dmp(t, transformation_system_func) for t in t_eval])

# Compute the forcing term values for plotting
f_fitted = np.array([rbf_forcing_temporal(t, N, weights, centers, widths, x_func, g, init[0]) for t in t_eval])

#######################
## ONLINE MODULATION ##
#######################

transformation_system_sol_mod = solve_ivp(lambda t, y: transformation_system(t, y, alpha_y, beta_y, g, lambda t: rbf_forcing_temporal(t, N, weights, centers, widths, lambda t: x_func_mod(t, x_func), g, init[0]), tau), t_span, init, t_eval=t_eval, dense_output=True)
transformation_system_func_mod = transformation_system_sol_mod.sol
y_dot_dot_traj_mod = np.array([y_dot_dot_dmp(t, transformation_system_func_mod) for t in t_eval])
x_traj_mod = np.array([x_func_mod(t, x_func) for t in t_eval])
f_fitted_mod = np.array([rbf_forcing_temporal(t, N, weights, centers, widths, lambda t: x_func_mod(t, x_func), g, init[0]) for t in t_eval])

##############
## PLOTTING ##
##############

# Plotting both the forcing term and the DMP trajectory (position and velocity)
plt.figure(figsize=(14, 7))
plt.title('Dynamic Movement Primitives')

# Plot Position
plt.subplot(5, 1, 1)
# plt.plot(t_eval, y_demo, label='demo', color='green', linestyle='--')
plt.plot(t_eval, transformation_system_sol.y[0], label='DMP', color='green', linestyle='-')
plt.plot(t_eval, transformation_system_sol_mod.y[0], label='DMP_mod', color='red', linestyle='-')
plt.axvline(x=0.65, color='grey', linestyle='-',)
plt.axvline(x=1.20, color='grey', linestyle='-',)
plt.xlabel('Time')
plt.ylabel('Position')
plt.grid()
plt.legend()

# Plot Velocity
plt.subplot(5, 1, 2)
# plt.plot(t_eval, y_dot_demo, label='demo', color='green', linestyle='--')
plt.plot(t_eval, transformation_system_sol.y[1], label='DMP', color='green', linestyle='-')
plt.plot(t_eval, transformation_system_sol_mod.y[1], label='DMP_mod', color='red', linestyle='-')
plt.axvline(x=0.65, color='grey', linestyle='-',)
plt.axvline(x=1.20, color='grey', linestyle='-',)
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.grid()
plt.legend()

# Plot Acceleration
plt.subplot(5, 1, 3)
# plt.plot(t_eval, y_dot_dot_demo, label='demo', color='green', linestyle='--')
plt.plot(t_eval, y_dot_dot_traj, label='DMP', color='green', linestyle='-')
plt.plot(t_eval, y_dot_dot_traj_mod, label='DMP_mod', color='red', linestyle='-')
plt.axvline(x=0.65, color='grey', linestyle='-',)
plt.axvline(x=1.20, color='grey', linestyle='-',)
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.grid()
plt.legend()

# Plot canonical system
plt.subplot(5, 1, 4)
plt.plot(t_eval, x_traj, label='x', color='green', linestyle='-')
plt.plot(t_eval, x_traj_mod, label='x_mod', color='red', linestyle='-')
plt.axvline(x=0.65, color='grey', linestyle='-',)
plt.axvline(x=1.20, color='grey', linestyle='-',)
plt.xlabel('Time')
plt.ylabel('Canonical system')
plt.grid()
plt.legend()

# Plot the forcing term
plt.subplot(5, 1, 5)
# plt.plot(t_eval, f_target, label='f_target', color='green', linestyle='--')
plt.plot(t_eval, f_fitted, label='f_fitted', color='green', linestyle='-')
plt.plot(t_eval, f_fitted_mod, label='f_fitted_mod', color='red', linestyle='-')
plt.axvline(x=0.65, color='grey', linestyle='-',)
plt.axvline(x=1.20, color='grey', linestyle='-',)
plt.xlabel('Time')
plt.ylabel('Forcing Term')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()
