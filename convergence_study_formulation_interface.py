# ------------------------------ LOAD LIBRARIES -------------------------------
from ngsolve import *
from phase_field import *
from sneddon import *
from meshes import make_crack_mesh
import pickle

SetNumThreads(4)


# -------------------------------- PARAMETERS ---------------------------------
h0 = 2
Lx = 6

file_out = 'sneddon_convergence_formulation_interface'


# ----------------------------------- DATA ------------------------------------
data = {'l0': 0.2, 'G_c': 5e2, 'E': 1e5, 'nu_s': 0.35, 'p': 4.5e3}
lines = [0, 0.13]
cod_ex = cod_exact(data, lines)
tcv_ex = tcv_exact(data)


# ------------------------ CHECK FOR PREVIOUS RESULTS -------------------------
try:
    with open(file_out + '.dat', 'rb') as fid:
        results = pickle.load(fid)
except FileNotFoundError:
    results = {'cod': {}, 'tcv': {}}


# ----------------------------- MESH CONVERGENCE ------------------------------
for lvl in range(Lx):
    if lvl in results['cod'].keys():
        continue

    print(f'lvl = {lvl}')

    # Construct mesh
    hmax = h0 * 0.5**lvl
    h_crack = hmax / 100
    with TaskManager():
        mesh = make_crack_mesh((-2, -2), (4, 4), -data['l0'], data['l0'], hmax,
                               h_crack)

    data_pf = {'eps': 0.5 * sqrt(h_crack), 'gamma': 100 * h_crack**-2,
               'kappa': 1e-10, 'n_steps': 5}

    pf_solver = phase_field_crack_solver(mesh=mesh, data_pf=data_pf, data=data,
                                         pre=data['p'], order=1)

    with TaskManager():
        # Compute Phase-Field Problem
        pf_solver.initialize_phase_field()
        gfu = pf_solver.solve_phase_field()

        # Post-process results
        cod = cod_from_phase_field(gfu, lines)
        tcv = tcv_from_phase_field(gfu)

    results['cod'][lvl] = ([(v[0], sqrt((v[1] - e[1])**2))
                            for v, e in zip(cod, cod_ex)])
    results['tcv'][lvl] = abs(tcv - tcv_ex)

    with open(file_out + '.dat', 'wb') as fid:
        pickle.dump(results, fid)


# -------------------------------- WRITE DATA ---------------------------------
for i, _x in enumerate(lines):
    file = file_out + f'_x{_x:.2f}.txt'
    with open(file, 'w') as fid:
        fid.write('lvl err\n')
        for lvl in range(Lx):
            fid.write(f'{lvl} {results["cod"][lvl][i][1]:5.3e}\n')


file = file_out + f'_tcv.txt'
with open(file, 'w') as fid:
    fid.write('lvl err\n')
    for lvl in range(Lx):
        fid.write(f'{lvl} {results["tcv"][lvl]:5.3e}\n')
