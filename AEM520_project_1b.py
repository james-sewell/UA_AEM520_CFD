import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

nx = 121
ny = 41

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


def build_b(b, rho, dt, u, v, dx, dy):
    
    b[1:-1, 1:-1] = (rho*(1/dt*((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx) + (v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy)) - ((u[1:-1, 2:] - u[1:-1, 0:-2])/(2*dx))**2 -
                    2*((u[2:, 1:-1] - u[0:-2, 1:-1])/(2*dy)*(v[1:-1, 2:] - v[1:-1, 0:-2])/(2*dx)) - ((v[2:, 1:-1] - v[0:-2, 1:-1])/(2*dy))**2))

    return b


def pressure_poisson(nt, p, dx, dy, b):
    pn = np.empty_like(p)
    pn = p.copy()
    nit = int(nt/1)
    
    for q in range(max_iter):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2])*dy**2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1])*dx**2)/(2*(dx**2 + dy**2)) -
                         dx**2*dy**2/(2*(dx**2 + dy**2))*b[1:-1,1:-1])

        p[:, 0] = 0         # p = 0 at x = 0
        p[0, :] = p[1, :]   # dp/dx = 0 at y = 0
        p[-1, :] = p[-2, :] # dp/dx =0 at y = 2
        p[:, -1] = 0        # p = 0 at x = 6

        err = (p - pn).max()
        if err < 1e-6:
            break
        
    return p


def pipe_flow(nt, u0, u, v, dt, dx, dy, p, rho, nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))
    ke = np.zeros((nt))
    max_p = np.zeros((nt))
    
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(nt, p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1]*dt/dx*(un[1:-1, 1:-1] - un[1:-1, 0:-2]) - vn[1:-1, 1:-1]*dt/dy*(un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                         dt/(2*rho*dx)*(p[1:-1, 2:] - p[1:-1, 0:-2]) + nu*(dt/dx**2*(un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                                                                           dt/dy**2*(un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] - un[1:-1, 1:-1]*dt/dx*(vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - vn[1:-1, 1:-1]*dt/dy*(vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt/(2*rho*dy)*(p[2:, 1:-1] - p[0:-2, 1:-1]) + nu*(dt/dx**2*(vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt/dy**2*(vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :]  = 0   # boundary layer at y = 0
        u[:, 0]  = u0   # set inlet velocity
        #u[:, -1] = 0
        u[-1, :] = 0   # boundary layer at y = 2
        v[0, :]  = 0
        #v[0, :] = v[1, :]  #dv/dy = 0 at y = 0
        v[-1, :] = 0
        #v[-1, :] = v[-2, :]  #dv/dy = 0 at y = 2
        v[:, 0]  = 0
        #v[:, -1] = 0

        #err_u = (u - un).max()
        #err_v = (v - vn).max()
        #if err_u < 1e-4 and err_v < 1e-4:
         #   break
        ke[n] = 0.5*(np.mean(np.array((u, v))))**2
        max_p[n] = np.max(p)
    
    return u, v, p, ke, max_p


u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
nt = 500
cfl = 0.2
u0 = 3.
dt = cfl*dx/u0
#dt = 0.001
Re = rho*u0*xl/nu
print('dt =', dt)
print('Re =', Re)

u, v, p, ke, max_p = pipe_flow(nt, u0, u, v, dt, dx, dy, p, rho, nu)

print(np.max(p))
print(np.shape(p))

fig = plt.figure(figsize=(13,7), dpi=100)
# plotting the pressure field as a contour
plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
plt.colorbar()
# plotting the pressure field outlines
plt.contour(X, Y, p, cmap=cm.viridis)  
# plotting velocity field
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
