from netgen.occ import SplineApproximation, MoveTo, OCCGeometry, Glue, X, Y
from ngsolve import Mesh


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


def make_fsi_mesh_from_points(corner, rectangle, points, hmax, hcrack, curvaturesafety):
    pts = points + [points[0]]
    fluid = SplineApproximation(pts, deg_min=1, deg_max=1).Face()
    fluid.faces.name = 'fluid'
    fluid.faces.maxh = hcrack
    fluid.edges.name = 'interface'

    base = MoveTo(*corner).Rectangle(*rectangle).Face()
    base.edges.Min(Y).name = 'bottom'
    base.edges.Max(X).name = 'right'
    base.edges.Max(Y).name = 'top'
    base.edges.Min(X).name = 'left'
    base -= fluid
    base.faces.name = 'solid'
    base.faces.maxh = hmax

    geo = OCCGeometry(Glue([base, fluid]), dim=2)
    ngmesh = geo.GenerateMesh(grading=0.2, curvaturesafety=curvaturesafety)

    return Mesh(ngmesh)
