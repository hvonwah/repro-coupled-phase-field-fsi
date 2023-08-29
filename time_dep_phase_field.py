# ------------------------------ LOAD LIBRARIES -------------------------------
from ngsolve import *
from meshes import make_crack_mesh, make_fsi_mesh_from_points
from phase_field import phase_field_crack_solver, cod_from_phase_field
from stokes_fluid_structure_interaction import stationary_stokes_fsi,\
    compute_mean_pressure
import numpy as np

SetNumThreads(4)


# -------------------------------- PARAMETERS ---------------------------------
t_end = 1
dt = 0.1

order_pf = 1
order_fsi = 2
hmax = 0.4
h_crack = hmax / 100
eps_cod = 0.5 * h_crack
n_itterations = 3

data_pf = {'eps': 0.04, 'gamma': 100 * h_crack**-2, 'kappa': 1e-10,
           'n_steps': 5}

vtk_flag = False
file_out = f'time-dep-fully-coupled'


# ----------------------------------- DATA ------------------------------------
data = {'l0': 0.2, 'G_c': 5e2, 'E': 1e5, 'nu_s': 0.35, 'p': 4.5e3, 'c_p': 2e2}

rhs_data_fluid = CF((1, 0))

mus = data['E'] / (2 * (1 + data['nu_s']))
ls = data['E'] * data['nu_s'] / ((1 + data['nu_s']) * (1 - 2 * data['nu_s']))
data_mat = {'rhof': 1e3, 'nuf': 0.1, 'mus': mus, 'lams': ls}


# ---------------------------- COMPUTE COD POINTS -----------------------------
def cod_lines(l0, d0, h):
    n = int(l0 / h / 2)
    _c = 0
    A = np.array([[1, 1], [n**2, n]])
    b = np.array([0, l0 + d0])
    _x = np.linalg.solve(A, b)
    lines = [_c + l0 + d0 - _x[0] * i**2 - _x[1] * i for i in range(1, n + 1)]
    lines2 = [2 * _c - lines[i] for i in range(n)]
    lines.reverse()
    return lines2[:-1] + lines


# ------------------------------- VISUALISATION -------------------------------
def phase_field_vtk(gfu, gf_pre_pf, it, time):
    gf_u, gf_phi = gfu.components
    mesh = gf_u.space.mesh
    vtk = VTKOutput(ma=mesh, coefs=[gf_u, gf_phi, gf_pre_pf],
                    names=['u', 'phi', 'p'],
                    filename=f'{file_out}pf_h{hmax}_{it}',
                    floatsize='single')
    vtk.Do(time=time)
    return None


def fsi_vtk(gfu, it, time):
    gf_vel, gf_pre, gf_deform = gfu.components
    mesh = gf_vel.space.mesh
    vtk = VTKOutput(ma=mesh, coefs=[gf_vel, gf_pre, gf_deform],
                    names=['vel', 'pre', 'def'],
                    filename=f'{file_out}fsi_h{hmax}_{it}', floatsize='single',
                    order=2)
    vtk.Do(time=time)
    return None


# ------------------------------- WRITE OUTPUT --------------------------------
file_crack_tips = f'{file_out}tips_h{hmax}.txt'

with open(file_crack_tips, 'w') as fid:
    fid.write('time tip-left tip-right\n')


# -------------------------------- MAIN LOOP ----------------------------------
l0 = -data['l0']
l1 = data['l0']
l0_last, l1_last = l0, l1
l0_new, l1_new = 0, 0

pre_vox = 0

with TaskManager():
    for i in range(int(ceil(t_end / dt) + 1)):
        t = i * dt
        print(f'\nt = {t:.6f}')

        # Sub-iteration between phase-field and FSI
        for j in range(n_itterations):
            # Phase-Field mesh
            print(f'              crack tips: {l0:.6f} {l1:.6f}')
            mesh = make_crack_mesh((-2, -2), (4, 4), l0, l1, hmax, h_crack)

            # Background and coupling pressure
            V_p = H1(mesh, order=order_pf)
            gf_pre_pf, gf_vox = GridFunction(V_p), GridFunction(V_p)

            gf_vox.Set(pre_vox)
            gf_pre_pf.Set(data['p'] + t * data['c_p'])
            gf_pre_pf.vec.data += gf_vox.vec

            # Set up and solve phase-field problem
            pf_solver = phase_field_crack_solver(
                mesh=mesh, data_pf=data_pf, data=data, pre=gf_pre_pf,
                order=order_pf, formulation=3, real_compile=True,
                wait_compile=False, info=True)

            pf_solver.initialize_phase_field()
            gfu_pf = pf_solver.solve_phase_field()
            gf_u, gf_phi = gfu_pf.components

            # Define points at which to compute COD
            lines = cod_lines(max(abs(l0), l1), 10 * h_crack, h_crack)
            cod = cod_from_phase_field(gfu_pf, lines)

            points = [(_c[0], _c[1] / 2) for _c in cod if _c[1] > eps_cod]
            l0_new, l1_new = points[0][0], points[-1][0]
            print(f'potential new crack tips: {l0_new:.6f} {l1_new:.6f}')
            if l0_new > l0_last:
                points = [(l0_last, 0)] + points
                l0 = l0_last
                n1 = 1
            else:
                l0 = l0_new
                n1 = 0
            if l1_new < l1_last:
                points = points + [(l1_last, 0)]
                l1 = l1_last
                n2 = -1
            else:
                l1 = l1_new
                n2 = len(points)
            points = points + [(p, -h) for p, h in reversed(points[n1:n2])]

            if j == n_itterations - 1:
                break

            # Construct FSI mesh
            mesh_fsi = make_fsi_mesh_from_points(
                (-2, -2), (4, 4), points, hmax=hmax, hcrack=h_crack,
                curvaturesafety=0.01)

            # Solve FSI problem
            gfu_fsi, f1, f2 = stationary_stokes_fsi(
                mesh=mesh_fsi, order=order_fsi, data=data_mat,
                compile_flag=True, rhs_data_fluid=rhs_data_fluid,
                newton_tol=1e-10, info=False)

            # Compute vertically averaged pressure
            pre_vox = compute_mean_pressure(
                gfu_fsi.components[1], xmin=l0, xmax=l1, h=h_crack)

        if l0 < l0_last:
            l0_last = l0
        if l1 > l1_last:
            l1_last = l1

        with open(file_crack_tips, 'a') as fid:
            fid.write(f'{t:.8f} {l0:.8f} {l1:.8f}\n')

        if vtk_flag:
            phase_field_vtk(gfu_pf, gf_pre_pf, i, t)
            fsi_vtk(gfu_fsi, i, t)
