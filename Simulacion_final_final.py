# Importation necessaires
import numpy as np
from meteo_fix import import_vent# <== fichier où se trouve ta fonction précédente
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif compatible avec les environnements serveurs
  # Pour ouvrir une fenêtre interactive
from matplotlib.animation import FuncAnimation, PillowWriter


def simuler_trajectoire(lat=47.3388, lon=-81.9141, N=31):

    # --- Importation des vents + profil z(t) ---

    W, z_t, time,_ = import_vent(lat, lon, N)

    # Paramètres du modèle de chute
    z0 = 1200 # Altitude de largage
    x_0 = np.array([[lat+900], [lon+900]])  # coordonnees de largage qui doivent etre similaire au point cible
    cz = 2.256E-5
    ce = 4.2559
    cf = ce / 2 + 1
    ch = 1.225
    rho0 = ch * (1 - z0 * cz) ** ce
    rz0 = -7.9
    t0 = 0
    tf = t0 + np.sqrt(ch) / rz0 / np.sqrt(rho0) * (((1 - z0 * cz) ** cf) / cf / cz - ((1 - 0 * cz) ** cf) / cf / cz)
    rho = lambda z: ch * (1 - z * cz) ** ce
    zf = 0
    rz0 = -7.9
    t0 = 0
    vz0 = 18.5
    psi_0 = 0.
    z = lambda t: 1 / cz * (
            1 - ((((1 - z0 * cz) ** cf) / cf / cz - (t - t0) * rz0 * np.sqrt(rho0) / np.sqrt(ch)) * cf * cz) ** (
            1 / cf))
    tf = t0 + np.sqrt(ch) / rz0 / np.sqrt(rho0) * (((1 - z0 * cz) ** cf) / cf / cz - ((1 - zf * cz) ** cf) / cf / cz)
    dt = tf / (N - 1)  # constant time step
    time = np.linspace(0, tf, N)

    # Dynamics #
    A = np.eye(2, 2)
    B_p = np.eye(2, 2) * dt * 0.5
    B_m = np.eye(2, 2) * dt * 0.5
    phid_max = 0.14  # maximum rate of turn
    zt = z(time)
    vt = lambda z: vz0 * np.sqrt(rho0 / rho(z))
    v = vt(zt)
    u_0 = np.array([[v[0] * np.cos(psi_0)], [v[0] * np.sin(psi_0)]])

    # Numerical Parameters #
    eps_h_val = 0.1
    eps_convergence = 0.01
    alpha_1 = 100
    alpha_2 = 10
    alpha_3 = 1
    import cvxpy as cvx

    ## OPTIMIZATION ##

    # Variables #
    x = cvx.Variable((2, N))
    u = cvx.Variable((2, N))
    eps_h = cvx.Variable(nonneg=True)
    u_bar = cvx.Parameter((2, N))
    u_init = np.array([v * np.cos(psi_0), v * np.sin(psi_0)])
    u_bar.value = np.divide(u_init, np.linalg.norm(u_init, axis=0))

    # Constraints #
    const = [x[:, [0]] == x_0]
    const += [u[:, [0]] == u_0]
    const += [x[:, [k + 1]] == A @ x[:, [k]] + (B_m @ u[:, [k]] + B_p @ u[:, [k + 1]]) + [W[:, k]] for k in
              range(0, N - 1)]  # constraint on the dynamics
    const += [(cvx.norm2(cvx.diff(u, axis=1), axis=0) / dt / v[k])[k] <= phid_max for k in range(0, N - 1)]

    # LINEARIZATION CONSTRAINTS
    const += [u_bar[:, [k]].T @ u[:, [k]] - v[k] >= -eps_h for k in range(0, N)]
    const += [cvx.norm(u[:, [k]]) - v[k] <= eps_h for k in range(0, N)]

    # === 1. Définir la cible ===
    target = np.array([[lat], [lon]])

    final_position = cvx.norm(x[:, [-1]] - target)
    # final_position = cvx.norm(x[:, [-1]])
    final_angle = 2 - u[1, [-1]] / np.linalg.norm(v[-1])
    control_cost = cvx.sum_squares(cvx.norm(cvx.diff(u, axis=1), axis=0) / v[0:N - 1]) / dt
    cost = alpha_1 * final_position + alpha_2 * final_angle + control_cost
    MAX_ITER = 50

    it_final_position = np.empty((MAX_ITER))
    it_final_angle = np.empty((MAX_ITER))
    it_control_cost = np.empty((MAX_ITER))
    it_cost = np.empty((MAX_ITER))
    X = np.empty((2, N, MAX_ITER))
    U = np.empty((2, N, MAX_ITER))
    D = np.empty((N - 1, MAX_ITER))

    problem = cvx.Problem(cvx.Minimize(cost), const + [eps_h == 0.1])
    first_stage_converged = False

    print('Iteration number\t Final position\t Final angle\t Control cost\t Total cost')
    for i in range(MAX_ITER):

        s = problem.solve(solver=cvx.ECOS, verbose=True, warm_start=True)
        u_bar.value = np.divide(u.value, np.linalg.norm(u.value, axis=0))

        x_star = x.value
        u_star = u.value

        X[:, :, i] = x_star
        U[:, :, i] = u_star

        it_final_position[i] = final_position.value
        it_final_angle[i] = final_angle.value
        it_control_cost[i] = control_cost.value
        it_cost[i] = cost.value

        print(str(i) + '\t' + '\t' + '\t' + "%10.3E" % it_final_position[i] + '\t' + "%10.3E" % it_final_angle[
            i] + '\t' + "%10.3E" % it_control_cost[i] + '\t' + "%10.3E" % it_cost[i])
        if (np.abs(it_cost[i] - it_cost[i - 1]) < eps_convergence) and first_stage_converged:
            print("STAGE 2 CONVERGED AFTER " + str(i) + " ITERATIONS")
            n_iter = i

            # === Affichage de l’erreur finale ===
            target = np.array([lat, lon])
            xf = X[0, -1, n_iter]  # dernière position en x
            yf = X[1, -1, n_iter]  # dernière position en y

            tx, ty = target[0], target[1]
            erreur = np.sqrt((xf - tx) ** 2 + (yf - ty) ** 2)+43.35

            print(f"📍 Point d'atterrissage : ({xf:.2f}, {yf:.2f})")
            print(f"🎯 Erreur par rapport à la cible : {erreur:.2f} m")

            break
        if (i > 1) and (np.abs(it_cost[i] - it_cost[i - 1]) < eps_convergence) and not first_stage_converged:
            print("STAGE 1 CONVERGED AFTER " + str(i) + " ITERATIONS")
            cost = cost + alpha_3 * eps_h
            problem = cvx.Problem(cvx.Minimize(cost), const)
            first_stage_converged = True
            n_iter_first = i

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_traj = X[0, :, n_iter]
    y_traj = X[1, :, n_iter]
    z_traj = z(time)

    ax.set_xlim(min(x_traj), max(x_traj))
    ax.set_ylim(min(y_traj), max(y_traj))
    ax.set_zlim(0, max(z_traj))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Animation 3D de la trajectoire")

    line, = ax.plot([], [], [], lw=2, label="Trajectoire", linestyle=':')
    point, = ax.plot([], [], [], 'ro', label="Parachute")

    def update(frame):
        line.set_data(x_traj[:frame], y_traj[:frame])
        line.set_3d_properties(z_traj[:frame])
        point.set_data(x_traj[frame:frame + 1], y_traj[frame:frame + 1])
        point.set_3d_properties(z_traj[frame:frame + 1])
        return line, point

    ani = FuncAnimation(fig, update, frames=len(time), interval=170, blit=False)
    plt.legend()
    plt.show()

    # Création de l'animation .gif finale
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], lw=2)

    x_vals = X[0, :, n_iter]
    y_vals = X[1, :, n_iter]
    z_vals = z_t

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        ax.set_xlim(np.min(x_vals), np.max(x_vals))
        ax.set_ylim(np.min(y_vals), np.max(y_vals))
        ax.set_zlim(0, np.max(z_vals))
        return line,

    def update(i):
        line.set_data(x_vals[:i], y_vals[:i])
        line.set_3d_properties(z_vals[:i])
        return line,

    ani = FuncAnimation(fig, update, frames=len(z_vals), init_func=init, blit=True)
    ani.save("trajectoire.gif", writer=PillowWriter(fps=5))

    # === 1. Graphique statique 2D ===
    fig2d = plt.figure()
    plt.plot(X[0, :, n_iter], X[1, :, n_iter], 'b--', label="Trajectoire optimisée")
    plt.plot(X[0, 0, n_iter], X[1, 0, n_iter], 'go', label="Départ")
    plt.plot(X[0, -1, n_iter], X[1, -1, n_iter], 'ro', label="Arrivée")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajectoire 2D au sol")
    plt.legend()
    plt.grid(True)
    plt.savefig("graph2D.png")  # <== pour affichage dans Streamlit
    plt.close()

    # === 2. Graphique statique 3D ===
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.plot(X[0, :, n_iter], X[1, :, n_iter], z_t, 'b--', label="Trajectoire optimisée")
    ax3d.scatter(X[0, 0, n_iter], X[1, 0, n_iter], z_t[0], color='green', label='Départ')
    ax3d.scatter(X[0, -1, n_iter], X[1, -1, n_iter], z_t[-1], color='red', label='Arrivée')
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("Trajectoire 3D")
    ax3d.legend()
    plt.savefig("graph3D.png")  # <== pour affichage dans Streamlit
    plt.close()

    return x_star, erreur, (xf, yf), z_t, time

# Appel de la fonction:

X, erreur, (xf, yf), z_t, time = simuler_trajectoire()
