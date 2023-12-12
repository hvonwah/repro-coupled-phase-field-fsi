from ngsolve import *
from ngsolve.solvers import Newton
from xfem import *
from xfem.mlset import *
import warnings


def pos(x):
    return IfPos(x, x, 0)


class phase_field_crack_solver():
    '''
    Class containing the tools and steps to compute a phase-field
    fracture model.

    Attributes
    ----------
    mesh : ngsolve.Mesh
        Computational mesh.
    pre : float /  ngsolve.CoefficientFuntion
        Pressure driving the fracture.
    eps : float
        Phase-field regularization parameter.
    gamma : float
        Penalty parameter.
    kappa : float
        Bulk regularisation parameter.
    n_steps : integer
        Number of loading steps.
    info : boolean
        Print info during phase-field computation
    gfu : ngsolve.GridFuntion
        Finite element function for the phase-field and displacement
        solutions.
    gfu_last : ngsolve.GridFuntion
        Store the solution from the last (loading) step
    a : ngsolve.BilinearForm
        Weak form of the phase-field problem PDE.
    inverse : string
        Sparse direct solver for linear systems

    Methods
    -------
    initialize_phase_field
        Initialize the phase-field solution according to the mesh
        materials.

    solve_phase_field
        Solve the Phase-field problem with given initial data.
    '''

    def __init__(self, mesh, data_pf, data, pre, order=1, bc_d='bottom|right|top|left', formulation=3, inverse='umfpack', real_compile=True, wait_compile=True, info=True, **kwargs):
        '''
        Set up phase-field solver.

        Parameters
        ----------
        mesh : ngsolve.Mesh
            Computational mesh
        data_pf : dict
            Phase-field discretisation parameters
        data : dict
            Material parameters
        pre : float / CoefficientFunction
            Pressure function
        order : int
            Polynomial order of the FE space. default = 1.
        bc_d : string
            Expression for the Dirichlet boundary condition for the phase-field
            deformation. default = 'bottom|right|top|left'.
        formulation : int
            Use bulk pressure coupling formulation (2) or interface
            pressure formulation (3).
        inverse : string
            Sparse direct solver to use for linear systems. Default = 'umfpack'
        real_compile : bool
            Hard compile integrators. default = True.
        wait_compile : bool
            Wait for hard compile to complete. Default = True.
        info : bool
            Print information to terminal.
        '''

        for key, value in kwargs.items():
            warnings.warn(f'Unknown keyword argument {key}={value} in'
                          + 'initiation of phase_field_crack_solver',
                          category=SyntaxWarning)

        self.eps = data_pf['eps']
        self.gamma = data_pf['gamma']
        self.kappa = data_pf['kappa']
        self.G_c = data['G_c']
        self.n_steps = data_pf['n_steps']

        self.info = info

        self.pre = pre
        self.mesh = mesh
        self.order = order
        self.inverse = inverse
        self.compile_opts = {'realcompile': real_compile, 'wait': wait_compile}

        if formulation not in [2, 3]:
            raise ValueError('formulation must be 2 or 3')

        mu_s = data['E'] / (2 * (1 + data['nu_s']))
        lam_s = data['nu_s'] * data['E'] / ((1 + data['nu_s']) * (1 - 2 * data['nu_s']))

        V1 = VectorH1(self.mesh, order=order, dirichlet=bc_d)
        V2 = H1(self.mesh, order=order)
        V = V1 * V2
        if self.info is True:
            print(f'Nr FreeDofs of PF space = {sum(V.FreeDofs())}')

        def e(U):
            return 1 / 2 * (Grad(U) + Grad(U).trans)

        def sigma(U):
            return 2 * mu_s * e(U) + lam_s * Trace(e(U)) * Id(self.mesh.dim)

        (u, phi), (v, psi) = V.TnT()

        self.gfu = GridFunction(V)
        self.gfu_last = GridFunction(V)

        phi_last = self.gfu_last.components[1]
        n = specialcf.normal(self.mesh.dim)

        form = ((1 - self.kappa) * phi_last**2 + self.kappa) * InnerProduct(sigma(u), e(v))
        form += (1 - self.kappa) * phi * InnerProduct(sigma(u), e(u)) * psi
        form += self.G_c * (- 1 / self.eps) * (1 - phi) * psi
        form += self.eps * InnerProduct(Grad(phi), Grad(psi))
        form += self.gamma * pos(phi - phi_last) * psi

        if formulation == 2:
            form += phi_last**2 * (self.pre * div(v) + InnerProduct(Grad(self.pre), v))
            form += 2 * phi * (self.pre * div(u) + InnerProduct(Grad(self.pre), u)) * psi

        form_bnd = - self.pre * n * v
        form_bnd += - 2 * self.pre * n * u * psi

        dx_interface = dx(definedon=self.mesh.Boundaries('interface'))

        self.a = BilinearForm(V, symmetric=False)
        self.a += form.Compile(**self.compile_opts) * dx

        if formulation == 3:
            self.a += form_bnd.Compile(**self.compile_opts) * dx_interface

        return None

    def initialize_phase_field(self):
        '''
        Initialize phase field with 0 in crack an 1 in solid.
        '''
        if self.info is True:
            print('Initialzie Phase-Field')

        V_loc = self.gfu.components[1].space
        freedofs = V_loc.FreeDofs()
        gf_phi, phi_last = GridFunction(V_loc), GridFunction(V_loc)

        gf_phi_inv = GridFunction(V_loc)
        gf_phi_inv.Set(1, definedon=self.mesh.Materials('crack'))
        gf_phi.Set(1 - gf_phi_inv)

        phi, psi = V_loc.TnT()
        form = self.G_c * (- 1 / self.eps) * (1 - phi) * psi
        form += self.eps * InnerProduct(Grad(phi), Grad(psi))
        form += self.gamma * pos(phi - phi_last) * psi

        a_loc = BilinearForm(V_loc)
        a_loc += form.Compile() * dx

        for i in range(self.n_steps):
            phi_last.vec.data = gf_phi.vec
            out = Newton(a_loc, gf_phi, freedofs=freedofs,
                         inverse=self.inverse, printing=False, maxerr=1e-8)
            if self.info is True:
                update = Norm(phi_last.vec - gf_phi.vec) / sum(freedofs)
                print(f'{i}, {out[1]:2d}, {update:.3e}')

        self.gfu.vec.data[:] = 0.0
        self.gfu.components[1].vec.data[:] = gf_phi.vec

        del a_loc, V_loc, gf_phi, phi_last, gf_phi_inv

        return None

    def solve_phase_field(self, u_d=None, dirichlet=None):
        '''
        Solve the phase-field problem with initial state stored in
        self.gfu.

        u_d : ngsolve.CoefficientFunktion
            If provided, the Dirichlet boundary condition for the deformation.
        dirichlet : string
            The names of the boundaries where the inhomogeneous Dirichlet
            boundary condition is to be applied.

        Returns
        -------
        self.gfu : ngsolve.GridFunction
            Finite element solution to phase-field problem.
        '''
        if self.info is True:
            print('Solve Phase-Field')

        if ((u_d is not None and dirichlet is None)
                or (u_d is None and dirichlet is not None)):
            raise ValueError('Both u_d and dirichlet have to be provided')
        if u_d is not None:
            gfu_u, gfu_pf = self.gfu_last.components
            gfu_u.Set(u_d, definedon=self.mesh.Boundaries(dirichlet))
            self.gfu.components[0].vec.data = gfu_u.vec

        freedofs = self.gfu.space.FreeDofs()

        for i in range(self.n_steps):
            self.gfu_last.vec.data = self.gfu.vec
            out = Newton(self.a, self.gfu, freedofs=freedofs,
                         inverse=self.inverse, printing=False, maxerr=1e-8)
            if self.info is True:
                update = Norm(self.gfu_last.vec - self.gfu.vec) / sum(freedofs)
                print(f'{i}, {out[1]:2d}, {update:.3e}')
        return self.gfu


def cod_from_phase_field(gfu, lines, vertical=False):
    '''
    Compute the crack opening displacements by integrating along a line
    through the phase-field.

    Parameters
    ----------
    gfu : ngsolve.GridFunction
        Finite element solution of the displacement and phase-field.
    lines : list
        List of x-Coordinates at which we integrate over the y-domain.

    Returns
    -------
    crack_openings : list[tuple]
        List of tuples containing the x-coordinate and the corresponding
        crack aperture.
    '''
    gf_u, gf_phi = gfu.components
    mesh = gf_u.space.mesh
    order = gf_u.space.globalorder

    _x = CF(x)
    if vertical is True:
        _x = CF(y)

    # Compute crack opening width based on phase-field
    lsetp1_line = GridFunction(H1(mesh, order=1))
    InterpolateToP1(_x - 2, lsetp1_line)
    ci_line = CutInfo(mesh, lsetp1_line)
    el_line = ci_line.GetElementsOfType(IF)
    ds_line = dCut(lsetp1_line, IF, order=2 * order, definedonelements=el_line)

    line_ind = InnerProduct(gf_u, Grad(gf_phi)).Compile()

    crack_openings = []
    for x0 in lines:
        InterpolateToP1(_x - x0, lsetp1_line)
        ci_line.Update(lsetp1_line)
        _cod = Integrate(line_ind * ds_line, mesh)

        crack_openings.append((x0, _cod))

    return crack_openings


def tcv_from_phase_field(gfu):
    '''
    Compute the total crack volume from a phase-field finite element
    solution.

    Parameters
    ----------
    gfu : ngsolve.GridFuntion
        Finite element Function of the displacements and phase field.

    Returns
    -------
    TVC : float
        Total crack volume predicted by solution.
    '''

    gf_u, gf_phi = gfu.components
    mesh = gf_u.space.mesh
    order = gf_u.space.globalorder
    return Integrate(InnerProduct(gf_u, Grad(gf_phi)).Compile(), mesh,
                     order=2 * order - 1)
