import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from findiff import FinDiff

nx = 121
ny = 41

#nx = 241
#ny = 81

xl = 6
yl = 2

max_iter = 1000

rho = 1.
nu = .1

dx = xl/(nx - 1)
dy = yl/(ny - 1)
x = np.linspace(0, xl, nx)
y = np.linspace(0, yl, ny)
X, Y = np.meshgrid(x, y)


def build_b(b, rho, dt, u, v, dx, dy, o):
    du_dx = FinDiff(1, dx, 1, acc=o)
    dv_dy = FinDiff(0, dy, 1, acc=o)
    du_dy = FinDiff(1, dy, 1, acc=o)
    dv_dx = FinDiff(0, dx, 1, acc=o)
    
    b[:, :] = rho*((du_dx(u) + dv_dy(v))/dt - du_dx(u)**2 - 2*du_dy(u)*dv_dx(v) - dv_dy(v)**2)

    #print(b)

    return b


def pressure_poisson(nt, p, dx, dy, b, o):
    pn = np.empty_like(p)
    pn = p.copy()
    nit = int(nt/1)
    
    for q in range(max_iter):
        pn = p.copy()
        d2p_dxdy = FinDiff((1, dx), (0, dy), acc=o)
        #p[:, :] = d2p_dxdy(pn) - dx**2*dy**2*b[:, :]/(2*(dx**2 + dy**2))
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2])*dy**2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1])*dx**2)/(2*(dx**2 + dy**2)) - dx**2*dy**2*b[1:-1,1:-1]/(2*(dx**2 + dy**2))
        
        #p[:, 0] = 0         # p = 0 at x = 0
        p[:, 0] = p[:, 1]
        p[0, :] = p[1, :]   # dp/dx = 0 at y = 0
        p[-1, :] = p[-2, :] # dp/dx =0 at y = 2
        #p[:, -1] = 0        # p = 0 at x = 6
        p[:, -1] = p[:, -2]

        #print(p)

        err = (p - pn).max()
        if err < 1e-6:
            break
        
    return p


def pipe_flow(nt, u0, u, v, dt, dx, dy, p, rho, nu, o):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))
    ke = np.zeros((nt))
    max_p = np.zeros((nt))
    mean_u = np.zeros((nt))
    mean_v = np.zeros((nt))
    conv_u = np.zeros((nt))
    conv_v = np.zeros((nt))
    du_dx = FinDiff(1, dx, 1, acc=o)
    dv_dy = FinDiff(0, dy, 1, acc=o)
    du_dy = FinDiff(1, dy, 1, acc=o)
    dv_dx = FinDiff(0, dx, 1, acc=o)
    dp_dx = FinDiff(1, dx, 1, acc=o)
    dp_dy = FinDiff(0, dy, 1, acc=o)
    d2u_dx2 = FinDiff(1, dx, 2, acc=o)
    d2u_dy2 = FinDiff(1, dy, 2, acc=o)
    d2v_dx2 = FinDiff(0, dx, 2, acc=o)
    d2v_dy2 = FinDiff(0, dy, 2, acc=o)
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_b(b, rho, dt, u, v, dx, dy, o)
        p = pressure_poisson(nt, p, dx, dy, b, o)
        
        #u[:, :] = un[:, :] - un[:, :]*dt*du_dx(un) - vn[:, :]*dt*du_dy(vn) - dt/(2*rho)*dp_dx(p) + nu*dt*(d2u_dx2(un) + d2u_dy2(un))

        #v[:, :] = vn[:, :] - un[:, :]*dt*dv_dx(vn) - vn[:, :]*dt*dv_dy(vn) - dt/(2*rho)*dp_dy(p) + nu*dt*(d2v_dx2(vn) + d2v_dy2(vn))

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1]*dt/dx*(un[1:-1, 1:-1] - un[1:-1, 0:-2]) - vn[1:-1, 1:-1]*dt/dy*(un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt/(2*rho*dx)*(p[1:-1, 2:] - p[1:-1, 0:-2]) + nu*(dt/dx**2*(un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         dt/dy**2*(un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] - un[1:-1, 1:-1]*dt/dx*(vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - vn[1:-1, 1:-1]*dt/dy*(vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt/(2*rho*dy)*(p[2:, 1:-1] - p[0:-2, 1:-1]) + nu*(dt/dx**2*(vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt/dy**2*(vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :]  = 0   # boundary layer at y = 0
        u[:, 0]  = u0   # set inlet velocity
        u[-1, :] = 0   # boundary layer at y = 2
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0

        mean_u[n] = np.mean(u)
        conv_u[n] = mean_u[n] - mean_u[n-1]
        mean_v[n] = np.mean(v)
        conv_v[n] = mean_v[n] - mean_v[n-1]
        ke[n] = 0.5*(np.mean(np.array((u, v))))**2
        max_p[n] = np.max(p)
        #if conv_u[n] and conv_v[n] < 1e-6:
        #    break
    
    return u, v, p, ke, max_p, conv_u, conv_v


u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
nt = 300
o = 8
cfl = 0.04
u0 = 16.7
dt = cfl*dx/u0
#dt = 0.0003
Re = rho*u0*xl/nu
print('dt =', dt)
print('Re =', Re)

u, v, p, ke, max_p, conv_u, conv_v = pipe_flow(nt, u0, u, v, dt, dx, dy, p, rho, nu, o)


fig = plt.figure(figsize=(13,7), dpi=100)
plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
plt.colorbar()
plt.contour(X, Y, p, cmap=cm.viridis)
plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

fig = plt.figure(figsize=(13, 7), dpi=100)
plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
plt.colorbar()
plt.contour(X, Y, p, cmap=cm.viridis)
plt.streamplot(X, Y, u, v)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

fig = plt.figure(figsize=(13, 7), dpi=100)
plt.plot(np.linspace(0, 1, nt), ke)
plt.title('Total Flow Kinetic Energy vs. time')
plt.xlabel('Timescale')
plt.ylabel('KE')
plt.show()

fig = plt.figure(figsize=(13, 7), dpi=100)
plt.plot(np.linspace(0, 1, nt), max_p)
plt.title('Flow Maximum Pressure vs. time')
plt.xlabel('Timescale')
plt.ylabel('p')
plt.show()

fig = plt.figure(figsize=(13, 7), dpi=100)
plt.plot(np.linspace(0, 1, nt), conv_u, label='u')
plt.plot(np.linspace(0, 1, nt), conv_v, label='v')
plt.legend()
plt.title('Velocity convergence')
plt.xlabel('Timescale')
plt.ylabel('velocites')
plt.show()
