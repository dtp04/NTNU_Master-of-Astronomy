import numpy as np
import matplotlib.pyplot as plt

class KeplerPropagator:
    def __init__(self, m0, m1, a, e):
        self.G = 4 * np.pi**2
        self.m0 = m0
        self.m1 = m1
        self.mu = self.G * (m0 + m1)
        self.a = a
        self.e = e
        self.n = np.sqrt(self.mu / a**3)

        r0_mag = a * (1 - e)
        v0_mag = np.sqrt(self.mu * (1 + e) / (a * (1 - e)))

        self.r = np.array([r0_mag, 0.0, 0.0])
        self.v = np.array([0.0, v0_mag, 0.0])
        self.E_curr = 0.0

        p0 = self.m1 * self.v
        self.l0 = np.cross(self.r, p0)
        self.energy0 = 0.5 * self.m1 * np.dot(self.v, self.v) - (self.G * self.m0 * self.m1) / r0_mag

    def solve_kepler_increment(self, dt, tol=1e-17):
        dm = self.n * dt
        de = dm

        for i in range(100):
            # ∆M = ∆E + (1 − cos ∆E) e sin E − (sin ∆E) e cos E
            f_val = de + (1 - np.cos(de)) * self.e * np.sin(self.E_curr) - \
                    np.sin(de) * self.e * np.cos(self.E_curr) - dm

            # Derivative: d/d(de)
            fp_val = 1 + np.sin(de) * self.e * np.sin(self.E_curr) - \
                     np.cos(de) * self.e * self.cos_En_calc()

            delta = f_val / fp_val
            de -= delta

            if abs(delta) < tol:
                return de, i
        return de, 100

    def cos_En_calc(self):
        return np.cos(self.E_curr)

    def step(self, dt):
        de, iterations = self.solve_kepler_increment(dt)
        r_mag = np.linalg.norm(self.r)

        f = 1 - (self.a / r_mag) * (1 - np.cos(de))
        g = dt - (1 / self.n) * (de - np.sin(de))

        r_next = f * self.r + g * self.v
        rn_mag = np.linalg.norm(r_next)

        f_dot = -(np.sqrt(self.mu * self.a) / (rn_mag * r_mag)) * np.sin(de)
        g_dot = 1 - (self.a / rn_mag) * (1 - np.cos(de))

        v_next = f_dot * self.r + g_dot * self.v

        self.r, self.v = r_next, v_next
        self.E_curr += de

        return self.r, self.v, iterations

system = KeplerPropagator(m0=1, m1=0.001, a=1, e=0.8)
dt = 0.01
total_time = 10
steps = int(total_time / dt)

traj = []
errors_e = []
errors_l = []

for _ in range(steps):
    r, v, _ = system.step(dt)
    traj.append(r.copy())

    p = system.m1 * v
    curr_energy = 0.5 * system.m1 * np.dot(v, v) - (system.G * system.m0 * system.m1) / np.linalg.norm(r)
    curr_l = np.cross(r, p)

    # error_E = (E - E') / E
    # error_l = |l - l'| / |l|
    errors_e.append((system.energy0 - curr_energy) / system.energy0)
    errors_l.append(np.linalg.norm(system.l0 - curr_l) / np.linalg.norm(system.l0))

traj = np.array(traj)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(traj[:, 0], traj[:, 1], color='black')
plt.plot(0, 0, 'o', label='Sun', color='yellow')
plt.axis('equal')
plt.title(f'Orbital Trajectory (e={system.e}, dt = {dt})')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.legend()
plt.grid(True, which='both', ls = '--')
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.semilogy(np.abs(errors_e), label='Relative Energy Error', c = 'black' , ls = '-')
plt.semilogy(errors_l, label='Relative Angular Momentum Error', ls='--', c = 'black')
plt.xlabel('Time Index')
plt.ylabel('Relative Error')
plt.title('Relative Error vs. Time Index')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

eccentricities = np.linspace(0.1,0.9,num=9)
dt = 0.2
total_time = 10
steps = int(total_time / dt)

plt.figure(figsize=(10, 8))
for e in eccentricities:
    system = KeplerPropagator(m0=1, m1=0.001, a=1, e=e)
    points = [system.r.copy()]
    for _ in range(steps):
        r, v, _ = system.step(dt)
        points.append(r.copy())
    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1], label=f'e = {e:.1f}')
plt.plot(0, 0, 'o', color='yellow', markersize=15, label='Sun')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc = 'upper left')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.title(f'Orbital Trajectories (dt = {dt})')
plt.tight_layout()
plt.show()

class KeplerPropagator:
    def __init__(self, m0, m1, a, e, method='newton'):
        self.G = 4 * np.pi**2
        self.m0, self.m1 = m0, m1
        self.mu = self.G * (m0 + m1)
        self.a, self.e = a, e
        self.n = np.sqrt(self.mu / a**3)
        self.method = method

        r0_mag = a * (1 - e)
        v0_mag = np.sqrt(self.mu * (1 + e) / (a * (1 - e)))

        self.r = np.array([r0_mag, 0.0, 0.0])
        self.v = np.array([0.0, v0_mag, 0.0])
        self.E_curr = 0.0

        p0 = self.m1 * self.v
        self.l0 = np.cross(self.r, p0)
        self.energy0 = 0.5 * self.m1 * np.dot(self.v, self.v) - (self.G * self.m0 * self.m1) / r0_mag

    def kepler_func(self, de, dm):
        return de + (1 - np.cos(de)) * self.e * np.sin(self.E_curr) - \
               np.sin(de) * self.e * np.cos(self.E_curr) - dm

    def solve_kepler_increment(self, dt, tol=1e-16):
        dm = self.n * dt

        if self.method == 'newton':
            de = dm
            for i in range(100):
                f_val = self.kepler_func(de, dm)
                fp_val = 1 + np.sin(de) * self.e * np.sin(self.E_curr) - \
                         np.cos(de) * self.e * np.cos(self.E_curr)
                delta = f_val / fp_val
                de -= delta
                if abs(delta) < tol: return de
            return de

        elif self.method == 'secant':
            de_prev, de = dm, dm * 1.01
            for i in range(100):
                f_prev = self.kepler_func(de_prev, dm)
                f_curr = self.kepler_func(de, dm)
                if abs(f_curr - f_prev) < 1e-18: break
                de_next = de - f_curr * (de - de_prev) / (f_curr - f_prev)
                if abs(de_next - de) < tol: return de_next
                de_prev, de = de, de_next
            return de

        elif self.method == 'bisection':
            low, high = dm - np.pi, dm + np.pi
            for i in range(100):
                mid = (low + high) / 2
                if self.kepler_func(low, dm) * self.kepler_func(mid, dm) < 0:
                    high = mid
                else:
                    low = mid
                if (high - low) / 2 < tol: return mid
            return (low + high) / 2

    def step(self, dt):
        de = self.solve_kepler_increment(dt)
        r_mag = np.linalg.norm(self.r)
        f = 1 - (self.a / r_mag) * (1 - np.cos(de))
        g = dt - (1 / self.n) * (de - np.sin(de))
        r_next = f * self.r + g * self.v
        rn_mag = np.linalg.norm(r_next)
        f_dot = -(np.sqrt(self.mu * self.a) / (rn_mag * r_mag)) * np.sin(de)
        g_dot = 1 - (self.a / rn_mag) * (1 - np.cos(de))
        v_next = f_dot * self.r + g_dot * self.v
        self.r, self.v = r_next, v_next
        self.E_curr += de
        return self.r, self.v

dt = 0.01
total_time = 10
steps = int(total_time / dt)
methods = ['newton', 'secant', 'bisection']
all_results = {}

for m in methods:
    system = KeplerPropagator(m0=1, m1=0.001, a=1, e=0.7, method=m)
    traj, err_e, err_l = [], [], []
    for _ in range(steps):
        r, v = system.step(dt)
        traj.append(r.copy())
        curr_energy = 0.5 * system.m1 * np.dot(v, v) - (system.G * system.m0 * system.m1) / np.linalg.norm(r)
        curr_l = np.cross(r, system.m1 * v)
        err_e.append(abs((system.energy0 - curr_energy) / system.energy0))
        err_l.append(np.linalg.norm(system.l0 - curr_l) / np.linalg.norm(system.l0))
    all_results[m] = {'traj': np.array(traj), 'err_e': err_e, 'err_l': err_l}

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(all_results['newton']['traj'][:, 0], all_results['newton']['traj'][:, 1], c='black')
plt.plot(0, 0, 'o', markersize=10, label='Sun', c='yellow')
plt.axis('equal')
plt.title(f'Orbital Trajectory (e={system.e}, dt={dt})')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.legend()
plt.grid(True, which='both', ls = '--')
plt.tight_layout()

plt.subplot(1, 2, 2)
colors = {'newton': 'green', 'secant': 'blue', 'bisection': 'red'}
for m in methods:
    plt.semilogy(all_results[m]['err_e'], color=colors[m], ls='-')
    plt.semilogy(all_results[m]['err_l'], color=colors[m], ls='--')

plt.xlabel('Time Index')
plt.ylabel('Relative Error')
plt.title('Relative Error vs. Time Index')
legend_elements = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Energy Error'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Ang. Momentum Error'),
    Line2D([0], [0], color='green', lw=2, linestyle='-', label='Newton-Raphson'),
    Line2D([0], [0], color='blue', lw=2, linestyle='-', label='Secant'),
    Line2D([0], [0], color='red', lw=2, linestyle='-', label='Bisection'),
]

plt.legend(handles=legend_elements)
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()