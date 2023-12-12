# ------------------------------ LOAD LIBRARIES -------------------------------
from ngsolve import *
from meshes import make_crack_mesh, make_fsi_mesh_from_points
from phase_field import phase_field_crack_solver, cod_from_phase_field
from stokes_fluid_structure_interaction import stationary_stokes_fsi,\
    compute_mean_pressure
import numpy as np
import argparse

SetNumThreads(8)

parser = argparse.ArgumentParser(description='Solve a time-dependent coupled FSI-Phase-Field problem', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-hmax', '--mesh_size', type=float, default=0.4, help='Global mesh size. The Phase-Field crack has mesh size smaller by a factor 100.')
parser.add_argument('-dt', '--time_step', type=float, default=0.1, help='Time step used')
parser.add_argument('-vtk', '--vtk_flag', type=int, default=0, help='Write VTK files. Enter 1 for True and 0 for False')
parser.add_argument('-eps', '--eps_cod', type=float, default=0.04)
parser.add_argument('-f', '--force', type=float, default=1)
parser.add_argument('-cp', '--c_pre', type=float, default=200)
options = vars(parser.parse_args())
print(options)


# -------------------------------- PARAMETERS ---------------------------------
t_end = 0.5                                     # Final time to compute to
dt = options['time_step']                       # Time-step

order_pf = 1                                    # Poly. order for phase field
order_fsi = 2                                   # Poly. order for FSI velocity
hmax = options['mesh_size']                     # Global mesh size
h_crack = hmax / 100                            # Phase field crack mesh size
eps = options['eps_cod']
gamma = 100 * h_crack**-2
eps_cod = 0.1 * h_crack                         # Phase field regularization
pf_iter = 5                                     # Phase field loading steps
kappa = 1e-10                                   # Bulk regularization parameter
sub_iter = 5                                    # PF-FSI sub-iterations
inverse = 'pardiso'                             # Sparse direct solver
real_compile = True                             # C++ coefficient functions
wait_compile = True                             # Wait for compile to complete
info = False                                    # Print info to console

vtk_flag = bool(options['vtk_flag'])            # Write vtk files of solution
file_out = f'example_4_time-dep-fully-coupled'  # File name for outputs
file_out += f'_h{hmax}dt{dt}eps{eps}'
file_out += f'_f{options["force"]}cp_{options["c_pre"]}'

# ----------------------------------- DATA ------------------------------------
data = {'l0': 0.2, 'G_c': 5e2, 'E': 1e5, 'nu_s': 0.35, 'p': 4.5e3,
        'c_p': options['c_pre']}

rhs_data_fluid = CF((options['force'], 0))

mus = data['E'] / (2 * (1 + data['nu_s']))
ls = data['E'] * data['nu_s'] / ((1 + data['nu_s']) * (1 - 2 * data['nu_s']))
data_mat = {'rhof': 1e3, 'nuf': 0.1, 'mus': mus, 'lams': ls}

data_pf = {'eps': eps, 'gamma': gamma, 'kappa': kappa, 'n_steps': pf_iter}


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
                    filename=f'results/vtk/{file_out}_pf{it}',
                    floatsize='single')
    vtk.Do(time=time)
    return None


def fsi_vtk(gfu, it, time):
    gf_vel, gf_pre, gf_deform, gf_lagr = gfu.components
    mesh = gf_vel.space.mesh
    vtk = VTKOutput(ma=mesh, coefs=[gf_vel, gf_pre, gf_deform],
                    names=['vel', 'pre', 'def'],
                    filename=f'results/vtk/{file_out}_fsi{it}',
                    floatsize='single', order=2)
    vtk.Do(time=time)
    return None


# ------------------------------- WRITE OUTPUT --------------------------------
file_crack_tips = f'results/{file_out}_tips.txt'

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
        for j in range(sub_iter):
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
                order=order_pf, formulation=3, inverse=inverse,
                real_compile=real_compile, wait_compile=wait_compile,
                info=info)

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

            if j == sub_iter - 1:
                if vtk_flag is True:
                    phase_field_vtk(gfu_pf, gf_pre_pf, i, t)
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

            if j == sub_iter - 2 and vtk_flag is True:
                fsi_vtk(gfu_fsi, i, t)

            # Compute vertically averaged pressure
            pre_vox = compute_mean_pressure(
                gfu_fsi.components[1], xmin=l0, xmax=l1, h=h_crack)

            del mesh_fsi, gfu_fsi, mesh, gf_pre_pf, gf_vox, points, \
                gf_u, gf_phi, gfu_pf, pf_solver

        if l0 < l0_last:
            l0_last = l0
        if l1 > l1_last:
            l1_last = l1

        with open(file_crack_tips, 'a') as fid:
            fid.write(f'{t:.8f} {l0:.8f} {l1:.8f}\n')
