from ngsolve import *
from ngsolve.solvers import Newton
from xfem import *
from xfem.mlset import *
import numpy as np


def stationary_stokes_fsi(mesh, order, data, rhs_data_fluid, bc_d='left|bottom|right|top', u_d_ih=None, bc_d_ih=None, alpha_u=1e-14, harmonic_extension=True, compile_flag=False, info=True, newton_damp=1, newton_tol=1e-14, inverse='pardiso', condense=False):
    '''
    Compute a stationary Stokes fluid-structure interaction problem.

    Parameters
    ----------
    mesh : ngsolve.Mesh
        Mesh with a 'fluid' and 'solid' regions, the boundary condition
        marker 'interface' between them and boundary marker 'out' for
        the outer homogeneous Dirichlet condition of the solid.
    order : int
        Order for the velocity and displacement finite element spaces.
    data : dict
        Dictionary containing the fluid and solid parameters.
    rhs_data_fluid : ngsolve.CoefficientFunction
        Right-hand side data for the fluid problem.
    bc_d : string
        Names of all Dirichlet boundaries.
    u_d_ih : ngsolve.CoefficientFunction
        If provided, the inhomogeneous Dirichlet boundary condition for
        the velocity.
    bc_d_ih : string
        If given, the part of the boundary where the inhomogeneous
        condition is to be applied.
    alpha_u : float
        Harmonic extension parameter
    harmonic_extension : bool
        Use harmonic extension for displacements into fluid.
    compile_flag : bool
        Compile cpp code.
    info : bool
        Print information to console.
    newton_damp : float
        Factor for Newton damping.
    newton_tol : float
        Residual tolerance for Newton solver.
    inverse : string
        Direct solver to use in Newton scheme.
    condense : bool
        Apply static condensation of internal bubbles for higher-order
        elements.

    Returns
    -------
    gfu, drag, lift : tuple(ngsolve.GridFunction, float, float)
        The FSI solution, drag and lift acting on the interface.
    '''
    mus = data['mus']
    ls = data['lams']
    rhof = data['rhof']
    nuf = data['nuf']

    # ------------------------- FINITE ELEMENT SPACE --------------------------
    V = VectorH1(mesh, order=order, dirichlet=bc_d)
    Q = H1(mesh, order=order - 1, definedon='fluid')
    D = VectorH1(mesh, order=order, dirichlet=bc_d)
    N = NumberSpace(mesh, definedon='fluid')
    X = V * Q * D * N
    (u, p, d, lam), (v, q, w, mu) = X.TnT()
    Y = V * Q * N
    (_u, _p, _lam), (_v, _q, _mu) = Y.TnT()

    gfu = GridFunction(X)
    if info:
        print(f'Nr FreeDofs of FSI space = {sum(X.FreeDofs(condense))}')

    # --------------------------- (BI)LINEAR FORMS ----------------------------
    Id2 = Id(2)
    F = Grad(d) + Id2
    C = F.trans * F
    E = 0.5 * (Grad(d) + Grad(d).trans)
    J = Det(F)
    Finv = Inv(F)
    FinvT = Finv.trans

    stress_sol = 2 * mus * E + ls * Trace(E) * Id2
    stress_fl = rhof * nuf * (grad(u) * Finv + FinvT * grad(u).trans)

    diff_fl = rhof * nuf * InnerProduct(J * stress_fl * FinvT, grad(v))
    pres_fl = -J * (Trace(grad(v) * Finv) * p + Trace(grad(u) * Finv) * q)
    # pres_fl += - J * 1e-9 * p * q
    pres_fl += - J * lam * q - J * mu * p

    rhs_fl = - InnerProduct(rhof * J * rhs_data_fluid, v)

    mass_sol = - InnerProduct(u, w)
    el_sol = InnerProduct(J * stress_sol * FinvT, grad(v))

    if harmonic_extension:
        extension = alpha_u * InnerProduct(Grad(d), Grad(d))
    else:
        gfdist = GridFunction(H1(mesh, order=1, dirichlet=bc_d))
        gfdist.Set(1, definedon=mesh.Boundaries('interface'))

        def NeoHookExt(C, mu=1, lam=1):
            ext = 0.5 * mu * Trace(C - Id2)
            ext += 0.5 * mu * (T2 * mu / lam * Det(C)**(-lam / 2 / mu) - 1)
            return ext

        extension = 1 / (1 - gfdist + 1e-2) * 1e-8 * NeoHookExt(C)

    stokes = nuf * rhof * InnerProduct(grad(_u), grad(_v))
    stokes += - div(_u) * _q - div(_v) * _p - _lam * _q - _mu * _p

    # ------------------------------ INTEGRATORS ------------------------------
    comp_opt = {'realcompile': compile_flag, 'wait': True}
    dFL, dSL = dx('fluid'), dx('solid')

    a = BilinearForm(X, symmetric=False, condense=condense)
    a += (diff_fl + pres_fl + rhs_fl).Compile(**comp_opt) * dFL
    a += (mass_sol + el_sol).Compile(**comp_opt) * dSL
    a += Variation(extension.Compile(**comp_opt) * dFL)

    a_stokes = BilinearForm(Y, symmetric=True, check_unused=False)
    a_stokes += stokes * dFL

    f_stokes = LinearForm(Y)
    f_stokes += InnerProduct(rhs_data_fluid, v) * dFL

    # ------------------------ FUNCTIONAL COMPUTATION -------------------------
    gfu_drag, gfu_lift = GridFunction(X), GridFunction(X)
    gfu_drag.components[0].Set(CF((1, 0)),
                               definedon=mesh.Boundaries('interface'))
    gfu_lift.components[0].Set(CF((0, 1)),
                               definedon=mesh.Boundaries('interface'))
    res = gfu.vec.CreateVector()

    # ----------------------------- SOLVE PROBLEM -----------------------------
    bts = Y.FreeDofs() & ~Y.GetDofs(mesh.Materials('solid'))
    bnd_dofs = V.GetDofs(mesh.Boundaries(f'out|interface|{bc_d}|{bc_d_ih}'))
    for i in range(V.ndof):
        if bnd_dofs[i]:
            bts[i] = False

    gfu_stokes = GridFunction(Y)
    res_stokes = gfu_stokes.vec.CreateVector()

    a_stokes.Assemble()
    f_stokes.Assemble()
    res_stokes.data = f_stokes.vec

    invstoke = a_stokes.mat.Inverse(bts, inverse=inverse)

    if u_d_ih is not None and bc_d_ih is not None:
        gfu_stokes.components[0].Set(u_d_ih, definedon=mesh.Boundaries(bc_d_ih))
        res_stokes.data -= a_stokes.mat * gfu_stokes.vec

    gfu_stokes.vec.data += invstoke * res_stokes

    gfu.components[0].vec.data = gfu_stokes.components[0].vec
    gfu.components[1].vec.data = gfu_stokes.components[1].vec

    Newton(a, gfu, maxit=10, inverse=inverse, maxerr=newton_tol,
           dampfactor=newton_damp, printing=info)

    a.Apply(gfu.vec, res)
    drag = - InnerProduct(res, gfu_drag.vec)
    lift = - InnerProduct(res, gfu_lift.vec)

    return gfu, drag, lift


def compute_mean_pressure(gf_pre, xmin, xmax, h, order=1):
    '''
    Compute a vertically averaged function between two points at
    1 / h samples.

    Parameters
    ----------
    gf_pre: GridFunction
        The function to average. Assumed to be defined in the mesh
        material fluid.
    xmin : float
        Left most point for samples
    xmax: float
        Right most point for sample.
    h : float
        Samples taken every interval of length (xmax - xmin) / h.
    order : int
        Integration order along vertical lines. default = 1.

    Returns
    -------
    Voxel Coefficient function.


    '''
    _n = int((xmax - xmin) / h)
    _h = (xmax - xmin) / _n
    lines = [xmin + i * _h for i in range(1, _n)]

    mesh = gf_pre.space.mesh

    lsetp1_line = GridFunction(H1(mesh, order=1))
    InterpolateToP1(x, lsetp1_line)
    ci_line = CutInfo(mesh, lsetp1_line)

    el_pre = BitArray(mesh.ne)
    el_pre.Clear()
    [el_pre.Set(el.nr) for el in mesh.Elements() if el.mat == "fluid"]

    ds_line = dCut(lsetp1_line, IF, order=order, definedonelements=el_pre)

    pre_mean = []
    cf_one = CF(1).Compile()
    for x0 in lines:
        InterpolateToP1(x - x0, lsetp1_line)
        ci_line.Update(lsetp1_line)
        val = Integrate(gf_pre * ds_line, mesh)
        val /= Integrate(cf_one * ds_line, mesh)
        pre_mean.append(val)

    values = np.array(pre_mean)
    values = values.reshape(1, len(pre_mean))
    return VoxelCoefficient((lines[0], -0.1), (lines[-1], 0.1), values,
                            linear=True)
