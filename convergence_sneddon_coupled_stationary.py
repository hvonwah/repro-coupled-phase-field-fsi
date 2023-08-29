# ------------------------------ LOAD LIBRARIES -------------------------------
from ngsolve import *
from meshes import *
from stokes_fluid_structure_interaction import *
from phase_field import *
import numpy as np
import pickle

SetNumThreads(4)


# -------------------------------- PARAMETERS ---------------------------------
h0 = 2
Lx = 6
order_pf = 1
order_fsi = 2

real_compile = True
wait_compile = True

vtk_flag = False

file_out = 'example2_'


# ----------------------------------- DATA ------------------------------------
n_itterations = 5
kappa = 1e-10

data = {'l0': 0.2, 'G_c': 5e2, 'E': 1e2, 'nu_s': 0.35, 'p': 4.5e0}

mus = data['E'] / (2 * (1 + data['nu_s']))
ls = data['E'] * data['nu_s'] / ((1 + data['nu_s']) * (1 - 2 * data['nu_s']))

e_a = data['l0']
e_b = 2 * (1 - data['nu_s']**2) * data['l0'] * data['p'] / data['E']

c1, c2 = 0.02, 1000
x0 = (e_a * 1 / 4, e_b * 2 / 3)
rhs_data_fluid = CF((c1 * exp(-c2 * ((x - x0[0])**2 + (y - x0[1])**2)), 0))

data_mat = {'rhof': 1e3, 'nuf': 0.1, 'mus': mus, 'lams': ls}


# ------------------------ CHECK FOR PREVIOUS RESULTS -------------------------
try:
    with open(file_out + 'convergence.dat', 'rb') as fid:
        results = pickle.load(fid)
except FileNotFoundError:
    results = {'cod': {}, 'tcv': {}}


# ---------------------------- COMPUTE COD POINTS -----------------------------
lines_conv = [0, 0.13]


def cod_lines(data, hmax, len_c):
    n = int(len_c / hmax / 4)
    A = np.array([[1, 1], [n**2, n]])
    b = np.array([4 * hmax, 0.22])
    _x = np.linalg.solve(A, b)

    lines = [_x[0] * i**2 + _x[1] * i for i in range(n + 1)]
    lines2 = [-lines[i] for i in range(n + 1)]
    lines2.reverse()
    return lines2[:-1] + lines


# ------------------------------- VISUALISATION -------------------------------
def phase_field_vtk(gfu, gf_pre_pf, lvl):
    gf_u, gf_phi = gfu.components
    mesh = gf_u.space.mesh
    vtk = VTKOutput(ma=mesh, coefs=[gf_u, gf_phi, gf_pre_pf],
                    names=['u', 'phi', 'p'], filename=f'{file_out}pf_lvl{lvl}',
                    floatsize='single')
    vtk.Do()
    return None


def fsi_vtk(gfu, lvl):
    gf_vel, gf_pre, gf_deform = gfu.components
    mesh = gf_vel.space.mesh
    vtk = VTKOutput(ma=mesh, coefs=[gf_vel, gf_pre, gf_deform],
                    names=['vel', 'pre', 'def'],
                    filename=f'{file_out}fsi_lvl{lvl}', floatsize='single',
                    order=2)
    vtk.Do()
    return None


# ----------------------------- MESH CONVERGENCE ------------------------------
for lvl in range(Lx):
    if lvl in results['cod'].keys():
        continue

    print(f'lvl = {lvl}')

    # Set up Phase-Field Problem
    hmax = h0 * 0.5**lvl
    h_crack = hmax / 100

    data_pf = {'eps': 0.5 * sqrt(h_crack), 'gamma': 100 * h_crack**-2,
               'kappa': 1e-10, 'n_steps': 5}

    with TaskManager():
        mesh = make_crack_mesh((-2, -2), (4, 4), -data['l0'], data['l0'],
                               hmax, h_crack)

    V_p = H1(mesh, order=order_pf)
    gf_pre_pf, gf_vox = GridFunction(V_p), GridFunction(V_p)

    pf_solver = phase_field_crack_solver(
        mesh=mesh, data_pf=data_pf, data=data, pre=gf_pre_pf, order=order_pf,
        real_compile=real_compile, wait_compile=wait_compile)

    lines = cod_lines(data, h_crack, 2 * data['l0'])
    n_cod, eps_cod = len(lines), 0.5 * h_crack

    collect_cod = []
    pre_vox = 0

    # Sub-iteration between Phase-Field and FSI
    for i in range(n_itterations):
        print(f'Itteration {i}')

        with TaskManager():
            # Set-up pressure
            gf_vox.Set(pre_vox)
            gf_pre_pf.Set(data['p'])
            gf_pre_pf.vec.data += gf_vox.vec

            # Phase-Field Problem
            pf_solver.initialize_phase_field()
            gfu = pf_solver.solve_phase_field()

            # Compute COD
            cod = cod_from_phase_field(gfu, lines)

        collect_cod.append(cod)
        points = [(cod[i][0], cod[i][1] / 2) for i in range(1, n_cod - 1)
                  if cod[i][1] > eps_cod]
        points = [(-data['l0'], 0)] + points + [(data['l0'], 0)]
        points = points + [(p, - h) for p, h in reversed(points[1:-1])]

        if i == n_itterations - 1:
            break

        # Compute FSI Problem
        with TaskManager():
            mesh_fsi = make_fsi_mesh_from_points((-2, -2), (4, 4), points,
                                                 hmax=hmax, hcrack=h_crack,
                                                 curvaturesafety=0.01)

            gfu_fsi, f1, f2 = stationary_stokes_fsi(
                mesh=mesh_fsi, order=order_fsi, data=data_mat,
                rhs_data_fluid=rhs_data_fluid)
            vel, pre, deform = gfu_fsi.components

            pre_vox = compute_mean_pressure(pre, xmin=-data['l0'],
                                            xmax=data['l0'], h=h_crack)

    # Compute convergence data
    with TaskManager():
        results['cod'][lvl] = cod_from_phase_field(gfu, lines_conv)
        results['tcv'][lvl] = tcv_from_phase_field(gfu)

    with open(file_out + 'convergence.dat', 'wb') as fid:
        pickle.dump(results, fid)

    # VTK Output for visualization
    if vtk_flag:
        with TaskManager():
            phase_field_vtk(gfu, gf_pre_pf, lvl)
            fsi_vtk(gfu_fsi, lvl)

    # Data output of crack-shape
    with open(f'{file_out}cod_lvl{lvl}.txt', 'w') as fid:
        str_out = 'pnt '
        for i in range(len(collect_cod)):
            str_out += f'cod{i} '
        fid.write(str_out + '\n')

        for i in range(n_cod):
            str_out = f'{collect_cod[0][i][0]} '
            for j in range(len(collect_cod)):
                str_out += f'{collect_cod[j][i][1]} '
            fid.write(str_out + '\n')


# -------------------------------- WRITE DATA ---------------------------------
file = file_out + f'cod.txt'
with open(file, 'w') as fid:
    str_out = 'lvl '
    for x in lines_conv:
        str_out += f'cod({x}) err({x}) '
    fid.write(str_out + '\n')
    for lvl in range(Lx):
        str_out = f'{lvl} '
        for i in range(len(lines_conv)):
            _cod = results["cod"][lvl][i][1]
            _err = abs(_cod - results["cod"][Lx - 1][i][1])
            str_out += f'{_cod:5.3e} {_err:5.3e} '
        fid.write(str_out + '\n')


file = file_out + f'tcv.txt'
with open(file, 'w') as fid:
    fid.write('lvl tcv err\n')
    for lvl in range(Lx):
        str_out = f'{lvl} {results["tcv"][lvl]:5.3e} '
        str_out += f'{abs(results["tcv"][lvl] - results["tcv"][Lx - 1]):5.3e}'
        fid.write(str_out + '\n')
