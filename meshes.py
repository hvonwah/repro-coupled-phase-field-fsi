from netgen.occ import SplineApproximation, MoveTo, OCCGeometry, Glue, X, Y
from ngsolve import Mesh, TaskManager


def make_crack_mesh(corner, rectangle, l0, l1, hmax, h_crack):
    base = MoveTo(*corner).Rectangle(*rectangle).Face()
    crack = MoveTo(l0, -h_crack)
    crack = crack.Rectangle(abs(l0) + l1, 2 * h_crack).Face()
    base -= crack

    base.faces.name = 'solid'
    base.faces.maxh = hmax
    base.edges.Min(Y).name = 'bottom'
    base.edges.Max(X).name = 'right'
    base.edges.Max(Y).name = 'top'
    base.edges.Min(X).name = 'left'
    crack.faces.name = 'crack'
    crack.faces.maxh = h_crack
    crack.faces.col = (1, 0, 0)
    crack.edges.name = 'interface'

    geo = OCCGeometry(Glue([base, crack]), dim=2)
    ngmesh = geo.GenerateMesh(grading=0.2)
    return Mesh(ngmesh)


def make_fsi_mesh_from_points(corner, rectangle, points, hmax, hcrack, curvaturesafety, inflow=False):
    base = MoveTo(*corner).Rectangle(*rectangle).Face()

    base.edges.Min(Y).name = 'bottom'
    base.edges.Max(X).name = 'right'
    base.edges.Max(Y).name = 'top'
    base.edges.Min(X).name = 'left'
    base.faces.name = 'solid'
    base.faces.maxh = hmax

    pts = points + [points[0]]
    fluid = SplineApproximation(pts, deg_min=1, deg_max=1).Face()
    fluid *= base
    base -= fluid

    fluid.faces.name = 'fluid'
    fluid.faces.maxh = hcrack
    fluid.edges.name = 'interface'
    if inflow:
        fluid.edges.Min(X).name = 'inflow'
    fluid.faces.col = (1, 0, 0)

    geo = OCCGeometry(Glue([base, fluid]), dim=2)
    ngmesh = geo.GenerateMesh(grading=0.2, curvaturesafety=curvaturesafety)

    return Mesh(ngmesh)


def make_side_crack_mesh(corner, rectangle, l1, hmax, h_crack):
    base = MoveTo(*corner).Rectangle(*rectangle).Face()
    crack = MoveTo(corner[0], -h_crack)
    crack = crack.Rectangle(l1, 2 * h_crack).Face()

    base.faces.name = 'solid'
    base.faces.maxh = hmax
    base.edges.Min(Y).name = 'bottom'
    base.edges.Max(X).name = 'right'
    base.edges.Max(Y).name = 'top'
    base.edges.Min(X).name = 'left'

    base -= crack

    crack.faces.name = 'crack'
    crack.faces.maxh = h_crack
    crack.faces.col = (1, 0, 0)
    crack.edges.Min(Y).name = 'interface'
    crack.edges.Max(X).name = 'interface'
    crack.edges.Max(Y).name = 'interface'
    crack.edges.Min(X).name = 'none'

    geo = OCCGeometry(Glue([base, crack]), dim=2)
    ngmesh = geo.GenerateMesh(grading=0.25)
    return Mesh(ngmesh)
