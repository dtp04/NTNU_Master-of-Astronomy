import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve

def acceleration_3d(x, y, z, vx, vy, vz, mu):
    mu1 = 1.0 - mu
    mu2 = mu
    r1 = np.sqrt((x + mu2)**2 + y**2 + z**2)
    r2 = np.sqrt((x - mu1)**2 + y**2 + z**2)
    ax = 2.0 * vy + x - (mu1 * (x + mu2) / r1**3) - (mu2 * (x - mu1) / r2**3)
    ay = -2.0 * vx + y - (mu1 * y / r1**3) - (mu2 * y / r2**3)
    az = - (mu1 * z / r1**3) - (mu2 * z / r2**3)

    return ax, ay, az

def get_lagrange_points(mu):
    mu1 = 1.0 - mu
    mu2 = mu
    L4 = (0.5 - mu2, np.sqrt(3.0) / 2.0, 0.0)
    L5 = (0.5 - mu2, -np.sqrt(3.0) / 2.0, 0.0)
    def collinear_eq(x):
        return x - (mu1 * (x + mu2) / np.abs(x + mu2)**3) - (mu2 * (x - mu1) / np.abs(x - mu1)**3)
    alpha = (mu2 / (3.0 * mu1))**(1.0 / 3.0)
    L1_x = fsolve(collinear_eq, mu1 - alpha)[0]
    L2_x = fsolve(collinear_eq, mu1 + alpha)[0]
    L3_x = fsolve(collinear_eq, -mu2 - 1.0)[0]

    return [(L1_x, 0.0, 0.0), (L2_x, 0.0, 0.0), (L3_x, 0.0, 0.0), L4, L5]

def leapfrog_step_3d(state, mu, dt):
    x, y, z, vx, vy, vz = state
    ax, ay, az = acceleration_3d(x, y, z, vx, vy, vz, mu)
    vx_half = vx + ax * (dt / 2.0)
    vy_half = vy + ay * (dt / 2.0)
    vz_half = vz + az * (dt / 2.0)

    x_new = x + vx_half * dt
    y_new = y + vy_half * dt
    z_new = z + vz_half * dt

    ax_new, ay_new, az_new = acceleration_3d(x_new, y_new, z_new, vx_half, vy_half, vz_half, mu)
    vx_new = vx_half + ax_new * (dt / 2.0)
    vy_new = vy_half + ay_new * (dt / 2.0)
    vz_new = vz_half + az_new * (dt / 2.0)

    return [x_new, y_new, z_new, vx_new, vy_new, vz_new]

def run_simulation(initial_state, mu, t_max, dt):
    steps = int(t_max / dt)
    history = np.zeros((steps, 3))
    state = list(initial_state)

    for step in range(steps):
        history[step, 0] = state[0]
        history[step, 1] = state[1]
        history[step, 2] = state[2]
        state = leapfrog_step_3d(state, mu, dt)

    return history

mu_baseline = 0.01
dt = 0.001
labels = ['L1', 'L2', 'L3', 'L4', 'L5']
L_pts = get_lagrange_points(mu_baseline)
delta_pos = 0.01
delta_vel = 0.01
z_baseline = 0.02

t_max=50
fig1 = plt.figure(figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')
for pt, name in zip(L_pts, labels):
    x0 = pt[0] + delta_pos
    state = [x0, pt[1], z_baseline, 0.0, 0.0, 0.0]
    traj = run_simulation(state, mu_baseline, t_max, dt)
    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"({x0:.4f},{name})")
    ax1.scatter([pt[0]], [pt[1]], [pt[2]], marker='x', color='black', s=40)

ax1.set_title("3D Orbits Under 5 Different x0 (mu = 0.01)")
ax1.set_xlabel("X (Synodic)")
ax1.scatter([-mu_baseline], [0.0], [0.0], color='black', marker='o', s=120, zorder=5, label='$m_1$')
ax1.scatter([1.0 - mu_baseline], [0.0], [0.0], color='black', marker='o', s=60, zorder=5, label='$m_2$')
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.set_ylabel("Y (Synodic)")
ax1.set_zlabel("Z (Synodic)")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

z_baseline = 0.0
fig1 = plt.figure(figsize=(7, 7))
ax1 = fig1.add_subplot(111)

for pt, name in zip(L_pts, labels):
    x0 = pt[0] + delta_pos
    state = [x0, pt[1], z_baseline, 0.0, 0.0, 0.0]
    traj = run_simulation(state, mu_baseline, t_max, dt)
    ax1.plot(traj[:, 0], traj[:, 1], label=f"({x0:.4f},{name})")
    ax1.scatter([pt[0]], [pt[1]], marker='x', color='black', s=40)

ax1.set_title("2D Orbits Under 5 Different x0 (mu = 0.01)")
ax1.scatter(-mu_baseline, 0.0, marker='o', color = 'black', s=120, label='$m_1$')
ax1.scatter(1.0 - mu_baseline, 0.0, marker='o', s=60, color = 'black', label='$m_2$')
ax1.set_xlabel("X (Synodic)")
ax1.set_ylabel("Y (Synodic)")
ax1.grid(True, alpha= 0.3, ls = ':')
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

t_max = 50
dt = 0.001

fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')
for pt, name in zip(L_pts, labels):
    y0 = pt[1] + delta_pos
    state = [pt[0], y0, z_baseline, 0.0, 0.0, 0.0]
    traj = run_simulation(state, mu_baseline, t_max, dt)
    ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"({name},{y0:.4f})")
    ax2.scatter([pt[0]], [pt[1]], [pt[2]], marker='x', color='black', s=40)

ax2.set_title("3D Orbits Under 5 Different y0 (mu = 0.01)")
ax2.set_xlabel("X (Synodic)")
ax2.set_ylabel("Y (Synodic)")
ax2.set_zlabel("Z (Synodic)")
ax2.scatter([-mu_baseline], [0.0], [0.0], color='black', marker='o', s=120, zorder=5, label='$m_1$')
ax2.scatter([1.0 - mu_baseline], [0.0], [0.0], color='black', marker='o', s=60, zorder=5, label='$m_2$')
#ax2.set_xlim(-1.5, 1.5)
#ax2.set_ylim(-1.5, 1.5)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

fig2 = plt.figure(figsize=(7, 7))
ax2 = fig2.add_subplot(111)

for pt, name in zip(L_pts, labels):
    y0 = pt[1] + delta_pos
    state = [pt[0], y0, 0.0, 0.0, 0.0, 0.0]
    traj = run_simulation(state, mu_baseline, t_max, dt)
    ax2.plot(traj[:, 0], traj[:, 1], label=f"({name},{y0:.4f})")
    ax2.scatter([pt[0]], [pt[1]], marker='x', color='black', s=40)

ax2.set_title("2D Orbits Under 5 Different y0 (mu = 0.01)")
ax2.set_xlabel("X (Synodic)")
ax2.set_ylabel("Y (Synodic)")
ax2.scatter([-mu_baseline], [0.0], color='black', marker='o', s=120, zorder=5, label='$m_1$')
ax2.scatter([1.0 - mu_baseline], [0.0], color='black', marker='o', s=60, zorder=5, label='$m_2$')
#ax2.set_xlim(-1.5, 1.5)
#ax2.set_ylim(-1.5, 1.5)
ax2.grid(True, alpha= 0.3, ls = ':')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

t_max=50
ref_pt = L_pts[3]
x0_fixed = ref_pt[0]
y0_fixed = ref_pt[1]
vy0_fixed = 0.0    # y_dot is kept identical at 0.0

vx0_values = [-0.05, -0.01, 0.01, 0.05]

fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111, projection='3d')
for vx0 in vx0_values:
    state = [x0_fixed, y0_fixed, z_baseline, vx0, vy0_fixed, 0.0]
    traj = run_simulation(state, mu_baseline, t_max, dt)
    ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"x0_dot = {vx0:.2f}")
ax3.scatter([ref_pt[0]], [ref_pt[1]], [ref_pt[2]], marker='x', color='black', s=40, label = f"L4")

ax3.set_title("3D Orbits Under Different x0dot (mu = 0.01)")
ax3.set_xlabel("X (Synodic)")
ax3.set_ylabel("Y (Synodic)")
ax3.set_zlabel("Z (Synodic)")
#ax3.set_xlim(-50, 50)
#ax3.set_ylim(-50, 50)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

fig3 = plt.figure(figsize=(7, 7))
ax3 = fig3.add_subplot(111)

for vx0 in vx0_values:
    state = [x0_fixed, y0_fixed, 0.0, vx0, vy0_fixed, 0.0]
    traj = run_simulation(state, mu_baseline, t_max, dt)
    ax3.plot(traj[:, 0], traj[:, 1], label=f"x0_dot = {vx0:.2f}")

ax3.scatter([ref_pt[0]], [ref_pt[1]], marker='x', color='black', s=40, label = f"L4")
ax3.set_title("2D Orbits Under Different x0dot (mu = 0.01)")
ax3.set_xlabel("X (Synodic)")
ax3.set_ylabel("Y (Synodic)")
ax3.grid(True, alpha= 0.3, ls = ':')
ax3.set_xlim(0.1,0.8)
ax3.set_ylim(0.5,1.2)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

t_max = 50
ref_pt = L_pts[3]
x0_fixed = ref_pt[0]
y0_fixed = ref_pt[1]
vx0_fixed = 0.0
vy0_values = [-0.05, -0.01, 0.01, 0.05]

fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111, projection='3d')

for vy0 in vy0_values:
    state = [x0_fixed, y0_fixed, z_baseline, vx0_fixed, vy0, 0.0]
    traj = run_simulation(state, mu_baseline, t_max, dt)
    ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"y0_dot = {vy0:.2f}")

ax3.scatter([ref_pt[0]], [ref_pt[1]], [ref_pt[2]], marker='x', color='black', s=40, label="L4")

ax3.set_title("3D Orbits Under Different y0dot (mu = 0.01)")
ax3.set_xlabel("X (Synodic)")
ax3.set_ylabel("Y (Synodic)")
ax3.set_zlabel("Z (Synodic)")
#ax3.set_xlim(0.2, 0.7)
#ax3.set_ylim(0.6, 1.1)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

fig3 = plt.figure(figsize=(7, 7))
ax3 = fig3.add_subplot(111)

for vy0 in vy0_values:
    state = [x0_fixed, y0_fixed, 0.0, vx0_fixed, vy0, 0.0]
    traj = run_simulation(state, mu_baseline, t_max, dt)
    ax3.plot(traj[:, 0], traj[:, 1], label=f"y0_dot = {vy0:.2f}")

ax3.scatter([ref_pt[0]], [ref_pt[1]], marker='x', color='black', s=40, label="L4")

ax3.set_title("2D Orbits Under Different y0dot (mu = 0.01)")
ax3.set_xlabel("X (Synodic)")
ax3.set_ylabel("Y (Synodic)")
ax3.grid(True, alpha=0.3, ls=':')
ax3.set_xlim(0.35, 0.65)
ax3.set_ylim(0.7, 1)
ax3.set_aspect('equal')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

dt = 0.001
t_max_stable = 150.0
t_max_unstable = 60.0
dv_nudge = 1e-4
mu_stable = 0.035
mu_unstable = 0.045

x0_s = 0.5 - mu_stable
y0_s = np.sqrt(3.0) / 2.0
init_state_s = [x0_s, y0_s, 0.0, dv_nudge, 0.0, 0.0]
traj_stable = run_simulation(init_state_s, mu_stable, t_max_stable, dt)

x0_u = 0.5 - mu_unstable
y0_u = np.sqrt(3.0) / 2.0
init_state_u = [x0_u, y0_u, 0.0, dv_nudge, 0.0, 0.0]
traj_unstable = run_simulation(init_state_u, mu_unstable, t_max_unstable, dt)

fig = plt.figure(figsize=(15, 6))

ax1 = fig.add_subplot(121)
dx_s = traj_stable[:, 0] - (0.5 - mu_stable)
dy_s = traj_stable[:, 1] - (np.sqrt(3.0) / 2.0)
ax1.plot(dx_s, dy_s, color='teal', label=f'Trajectory ($\mu$ = {mu_stable})')
ax1.scatter(0, 0, color='black', marker='X', s=100, zorder=5, label='$L_4$ Center')
ax1.set_title("Stable Regime ($\mu$ = 0.035 < $\mu_{limit}$)")
ax1.set_xlabel("$\Delta$X from $L_4$")
ax1.set_ylabel("$\Delta$Y from $L_4$")
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.axis('equal')
ax1.legend()

ax2 = fig.add_subplot(122)
dx_u = traj_unstable[:, 0] - (0.5 - mu_unstable)
dy_u = traj_unstable[:, 1] - (np.sqrt(3.0) / 2.0)
ax2.plot(dx_u, dy_u, color='crimson', label=f'Trajectory ($\mu$ = {mu_unstable})')
ax2.scatter(0, 0, color='black', marker='X', s=100, zorder=5, label='$L_4$ Center')
ax2.set_title("Unstable Regime ($\mu$ = 0.045 > $\mu_{limit}$)")
ax2.set_xlabel("$\Delta$X from $L_4$")
ax2.set_ylabel("$\Delta$Y from $L_4$")
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 5, figsize=(20, 10))
mu_values = [0.025, 0.035, 0.045]

for row, mu in enumerate(mu_values):
    L_points = get_lagrange_points(mu)

    for col, L in enumerate(L_points):
        ax = axes[row, col]
        nudge = 1e-4
        state = [L[0] + nudge, L[1] + nudge, 0, 0, 0, 0]
        t_max = 150.0 if row == 0 else 100.0
        traj = run_simulation(state, mu, t_max, dt)
        dx = traj[:, 0] - L[0]
        dy = traj[:, 1] - L[1]

        ax.plot(dx, dy)
        ax.scatter(0, 0, color='black', marker='X', s=100)
        ax.set_title(f"$\mu$={mu:.3f}, $L_{col+1}$")
        ax.set_xlabel("$\Delta$X")
        ax.set_ylabel("$\Delta$Y")
        ax.grid(True, linestyle=':', alpha=0.6)
        #ax.set_xlim(-0.1, 0.1)
        #ax.set_ylim(-0.1, 0.1)

plt.tight_layout()
plt.show()

from matplotlib.colors import ListedColormap
from scipy.optimize import fsolve

def get_U(x, y, mu):
    mu1 = 1.0 - mu
    mu2 = mu
    r1 = np.sqrt((x + mu2)**2 + y**2)
    r2 = np.sqrt((x - mu1)**2 + y**2)
    return 0.5*(x**2 + y**2) + mu1/r1 + mu2/r2

def get_lagrange_points(mu):
    mu1 = 1.0 - mu
    L4 = (0.5 - mu, np.sqrt(3)/2)
    L5 = (0.5 - mu, -np.sqrt(3)/2)

    def collinear_eq(x):
        return (x - mu1*(x + mu)/np.abs(x + mu)**3 - mu*(x - 1 + mu)/np.abs(x - 1 + mu)**3)

    L1 = (fsolve(collinear_eq, 1 - mu - 0.1)[0], 0)
    L2 = (fsolve(collinear_eq, 1 - mu + 0.1)[0], 0)
    L3 = (fsolve(collinear_eq, -mu - 1.0)[0], 0)

    return [L1, L2, L3, L4, L5]

def plot_regimes(mu=0.01):
    cj_test = [2.9, 3.00, 3.1, 3.15, 3.25, 3.35]
    m1 = (-mu, 0)
    m2 = (1 - mu, 0)
    l_points = get_lagrange_points(mu)
    x = np.linspace(-1.5, 1.5, 600)
    y = np.linspace(-1.5, 1.5, 600)
    X, Y = np.meshgrid(x, y)
    pot = get_U(X, Y, mu)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, cj in enumerate(cj_test):
        allowed = (2 * pot) >= cj
        axes[i].imshow( allowed,extent=[-1.5, 1.5, -1.5, 1.5],cmap=ListedColormap(['lightgray', 'white']),origin='lower')

        lx = [p[0] for p in l_points]
        ly = [p[1] for p in l_points]

        if i == 0:
            axes[i].scatter(m1[0], m1[1], color='blue', s=180, label=r'$m_1$')
            axes[i].scatter(m2[0], m2[1], color='blue', s=60, label=r'$m_2$')
            axes[i].scatter(lx, ly, color='black', marker='x', s=100, label='L-points')
        else:
            axes[i].scatter(m1[0], m1[1], color='blue', s=180)
            axes[i].scatter(m2[0], m2[1], color='blue', s=60)
            axes[i].scatter(lx, ly, color='black', marker='x', s=100)

        axes[i].set_title(f'$C_J={cj}$')
        axes[i].set_aspect('equal')
        axes[i].set_xlim(-1.5, 1.5)
        axes[i].set_ylim(-1.5, 1.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=True)

    plt.tight_layout()
    plt.show()

plot_regimes(0.01)

def get_lagrange_points(mu):
    mu1 = 1.0 - mu
    mu2 = mu
    L4 = (0.5 - mu2, np.sqrt(3.0) / 2.0, 0.0)
    L5 = (0.5 - mu2, -np.sqrt(3.0) / 2.0, 0.0)
    def collinear_eq(x):
        return x - (mu1 * (x + mu2) / np.abs(x + mu2)**3) - (mu2 * (x - mu1) / np.abs(x - mu1)**3)

    alpha = (mu2 / (3.0 * mu1))**(1.0 / 3.0)
    L1_x = fsolve(collinear_eq, mu1 - alpha)[0]
    L2_x = fsolve(collinear_eq, mu1 + alpha)[0]
    L3_x = fsolve(collinear_eq, -mu2 - 1.0)[0]
    return [(L1_x, 0.0, 0.0), (L2_x, 0.0, 0.0), (L3_x, 0.0, 0.0), L4, L5]

def cr3bp_derivs(t, state, mu):
    x, y, vx, vy = state
    mu1 = 1.0 - mu
    mu2 = mu
    r1 = np.sqrt((x + mu2)**2 + y**2)
    r2 = np.sqrt((x - mu1)**2 + y**2)
    ax = 2.0 * vy + x - (mu1 * (x + mu2) / r1**3) - (mu2 * (x - mu1) / r2**3)
    ay = -2.0 * vx + y - (mu1 * y / r1**3) - (mu2 * y / r2**3)
    return [vx, vy, ax, ay]

def simulate_orbit(mu, initial_state, t_span, max_step=0.1):
    sol = solve_ivp(
        cr3bp_derivs,
        t_span,
        initial_state,
        args=(mu,),
        method='Radau',
        rtol=1e-10,
        atol=1e-10,
        max_step=max_step
    )
    return sol.y[0], sol.y[1]

def plot_orbit(mu, x, y, title, init_state,xlim=None,ylim=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, color='blue')
    ax.scatter([-mu], [0], color='black', s=180, label='$m_1$')
    ax.scatter([1.0 - mu], [0], color='black', s=60, label='$m_2$')
    l_points = get_lagrange_points(mu)
    l_x = [p[0] for p in l_points]
    l_y = [p[1] for p in l_points]
    ax.scatter(l_x, l_y, color='red', marker='x', s=60, label='$L_{1-5}$')

    x0, y0, vx0, vy0 = init_state[0], init_state[1], init_state[2], init_state[3]
    legend_text = (
        f"$\mu = {mu}$\n"
        f"$x_0={x0:.3f}, y_0={y0:.3f}$\n"
        f"$\dot{{x}}_0={vx0}, \dot{{y}}_0={vy0}$")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend([legend_text], loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.show()

mu_tad = 0.001
state_tad = [0.499 + 0.0065, 0.866 + 0.0065, 0.0, 0.0]
x_tad, y_tad = simulate_orbit(mu_tad, state_tad, t_span=[0, 200])

plot_orbit(mu_tad, x_tad, y_tad, "Tadpole Orbit around $L_4$", state_tad)

mu_horse = 0.00095
state_horse = [-1.0275, 0.0, 0.0, 0.0403]
x_h, y_h = simulate_orbit(mu_horse, state_horse, t_span=[0, 3000])
plot_orbit(mu_horse, x_h, y_h, "Horseshoe Orbit", state_horse)

mu_bound = 0.01
state_bound = [1.05, 0.0, 0.0, -0.3]
x_b, y_b = simulate_orbit(mu_bound, state_bound, t_span=[0, 20])
plot_orbit(mu_bound, x_b, y_b, "Bounded Orbit around $m_2$",
           state_bound, xlim=[0.8, 1.2], ylim=[-0.2, 0.2])

mu_trans = 0.01
state_trans = [0.85, 0.0, 0.0, 0.25]
x_t, y_t = simulate_orbit(mu_trans, state_trans, t_span=[0, 150])
plot_orbit(mu_trans, x_t, y_t, "Transit Orbit through $L_1$", state_trans)

mu_chaos = 0.01
state_chaos = [1.15, 0.05, 0.0, 0.1]
x_c, y_c = simulate_orbit(mu_chaos, state_chaos, t_span=[0, 200])
plot_orbit(mu_chaos, x_c, y_c, "Chaotic Trajectory near $L_2$", state_chaos)

mu_esc = 0.01
state_esc = [1.0, 0.0, 0.0, 1.5]
x_e, y_e = simulate_orbit(mu_esc, state_esc, t_span=[0, 20])
plot_orbit(mu_esc, x_e, y_e, "Escape Trajectory", state_esc)

mu_res = 0.01
state_res = [0.45, 0.0, 0.0, 0.65]
x_r, y_r = simulate_orbit(mu_res, state_res, t_span=[0, 50])
plot_orbit(mu_res, x_r, y_r, "2:1 Resonant Orbit", state_res)