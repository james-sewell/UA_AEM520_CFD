"""

Momentum:           ∂u/∂t + (u ⋅ ∇) u = − 1/ρ ∇p + ν ∇²u + f

Incompressibility:  ∇ ⋅ u = 0


u:  Velocity (2d vector)
p:  Pressure
f:  Forcing (here =0)
ν:  Kinematic Viscosity
ρ:  Density
t:  Time
∇:  Nabla operator (defining nonlinear convection, gradient and divergence)
∇²: Laplace Operator

----

* Pressure have zero initial condition.
* Velocity will have an inlet value and zero for everywhere else
* Homogeneous Dirichlet Boundary Conditions everywhere except for inlet
  velocity at top. It is driven by an external flow.

-----

Solution strategy:

1. Solve Momentum equation without pressure gradient for tentative velocity
   (with given Boundary Conditions)

    ∂u/∂t + (u ⋅ ∇) u = ν ∇²u

2. Solve pressure poisson equation for pressure at next point in time
   (with homogeneous Neumann Boundary Conditions everywhere except for
   the outlet, where it is homogeneous Dirichlet)

    ∇²p = -ρ(∂u/∂x*∂u/∂x + 2(∂u/∂y * ∂v/∂x) + ∂v/∂y*∂v/∂y)           

3. Correct the velocities (and again enforce the Velocity Boundary Conditions)

    u ← u − Δt/ρ ∇ p

"""

import matplotlib.pyplot as plt
import numpy as np
ngc = 1
N_POINTS_x = 47
N_POINTS_y = 47
DOMAIN_SIZE_x = 1.0
DOMAIN_SIZE_y = 1.0
N_ITERATIONS = 9000
KINEMATIC_VISCOSITY = 0.01
DENSITY = 1.0
HORIZONTAL_VELOCITY_TOP = 0.75
RE = DENSITY*HORIZONTAL_VELOCITY_TOP*DOMAIN_SIZE_x/KINEMATIC_VISCOSITY
N_PRESSURE_POISSON_ITERATIONS = 5000
cfl = 0.09

########################################################################
#Object Inputs
if True:
    R_Object = 0.25
    Center_Object_x = 0.5
    Center_Object_y = 0.5
########################################################################

if __name__ == "__main__":
    dx = DOMAIN_SIZE_x / (N_POINTS_x - 1)
    dy = DOMAIN_SIZE_y / (N_POINTS_y - 1)
    x = np.linspace(0.0, DOMAIN_SIZE_x, N_POINTS_x)
    y = np.linspace(0.0, DOMAIN_SIZE_y, N_POINTS_y)
    X, Y = np.meshgrid(x, y)
    
    #Setup Object in Channel
    if True:
        X_object = np.round(R_Object*np.cos(np.linspace(0, 2*np.pi,75)) + Center_Object_x,8)
        Y_object = np.round(R_Object*np.sin(np.linspace(0, 2*np.pi,75)) + Center_Object_y,8)
        
    #Velocities & Pressure Initial Conditions
    u_prev = np.zeros_like(X)
    v_prev = np.zeros_like(X)
    p_prev = np.ones_like(X)
    
    #Declare Functions
    def Circle_BC(vector, X, Y, Center_Object_x, Center_Object_y, R_Object, Value = 0):
        # vector[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object + np.sqrt(dx**2 + dy**2)] = Value
        
        X_i = X[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object + np.sqrt(dx**2 + dy**2)]
        vector[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object + np.sqrt(dx**2 + dy**2)] = 0#Value*np.exp(-X_i)#vector[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object].mean()
        vector[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object] = 0#vector[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object].mean()
        return vector
    
    def Circle_BC_P(vector, X, Y, Center_Object_x, Center_Object_y, R_Object, Value = 0):
        X_i = X[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object + np.sqrt(dx**2 + dy**2)]
        
        #use e^-x to represent a decaying pressure with x
        vector[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object + np.sqrt(dx**2 + dy**2)] = Value*np.exp(-X_i)#vector[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object].mean()
        vector[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object] = 0#vector[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object].mean()
        return vector
    
    def V_x_BC(vector):
        vector[:, 0] = HORIZONTAL_VELOCITY_TOP#u_tent[:, 1]#HORIZONTAL_VELOCITY_TOP #Left
        vector[:, -1] = vector[:, -2] #Right
        vector[-1, :] = 0.0 #Top
        vector[0, :] = 0.0 #Bottom
        vector = Circle_BC(vector, X, Y, Center_Object_x, Center_Object_y, R_Object, 0)
        return vector
    
    def V_y_BC(vector):
        vector[0, :] = 0.0 #Bottom
        vector[:, 0] = 0#v_tent[:, 1] #Left
        vector[:, -1] = vector[:, -2] #Right
        vector[-1, :] = 0.0 #Top
        vector = Circle_BC(vector, X, Y, Center_Object_x, Center_Object_y, R_Object, 0)
        return vector

    def central_difference_x(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 2:  ]
            -
            f[1:-1, 0:-2]
        ) / (
            2 * dx
        )
        return diff
    
    def central_difference_y(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[2:  , 1:-1]
            -
            f[0:-2, 1:-1]
        ) / (
            2 * dy
        )
        return diff
    
    def POISSON_B_TERM(density, u, v, dx, dy):
        #b= -ρ(∂u/∂x*∂u/∂x + 2(∂u/∂y * ∂v/∂x) + ∂v/∂y*∂v/∂y)
        b = np.zeros_like(u)
        dudx = central_difference_x(u) 
        dudy = central_difference_y(u) 
        dvdx = central_difference_x(v) 
        dvdy = central_difference_y(v) 
        b[ngc:-ngc] = -density*(dudx[ngc:-ngc]*dudx[ngc:-ngc] + 2*(dudy[ngc:-ngc] * dvdx[ngc:-ngc]) + dvdy[ngc:-ngc]*dvdy[ngc:-ngc])
        return b
    
    def poisson_second_order(Pn, dx, dy, BC_R, BC_L, BC_T, BC_B, ngc, b):
        P = np.zeros_like(Pn)
        P[ngc:-ngc, ngc:-ngc] = ((dx**2)*(Pn[ngc+1:,ngc:-ngc]+Pn[:-ngc-1,ngc:-ngc]) + \
                                 (dy**2)*(Pn[ngc:-ngc,ngc+1:]+Pn[ngc:-ngc,:-ngc-1]) - \
                                 b[ngc:-ngc, ngc:-ngc]*(dx**2 * dy**2)) / (2 * (dx**2 + dy**2))
        
        P[0,:]          = BC_L*dx + P[1,:]
        P[-1, :] = BC_R*dx + P[-2,:]
        P[:,0]          = BC_B*dy + P[:,1]
        P[:, -1] = BC_T*dy - P[:,-2]
        P = Circle_BC_P(P, X, Y, Center_Object_x, Center_Object_y, R_Object, P.mean())
    
        return P
    
    def laplace(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = ((f[1:-1, 0:-2]+f[1:-1, 2:]-2*f[1:-1, 1:-1])/(dx**2)) + ((f[0:-2, 1:-1]+f[2:, 1:-1]-2*f[1:-1, 1:-1])/(dy**2))
        return diff
    
    #Begin Main
    for _ in range(N_ITERATIONS):
        print(_)
        d_u_prev__d_x = central_difference_x(u_prev)
        d_u_prev__d_y = central_difference_y(u_prev)
        d_v_prev__d_x = central_difference_x(v_prev)
        d_v_prev__d_y = central_difference_y(v_prev)
        laplace__u_prev = laplace(u_prev)
        laplace__v_prev = laplace(v_prev)
        
        
        dt = (min(dx, dy)/max(u_prev.max(),v_prev.max(), HORIZONTAL_VELOCITY_TOP))*cfl
        # Perform a tentative step by solving the momentum equation without the
        # pressure gradient
        u_tent = (
            u_prev
            +
            dt * (
                -
                (
                    u_prev * d_u_prev__d_x
                    +
                    v_prev * d_u_prev__d_y
                )
                +
                KINEMATIC_VISCOSITY * laplace__u_prev
            )
        )
        v_tent = (
            v_prev
            +
            dt * (
                -
                (
                    u_prev * d_v_prev__d_x
                    +
                    v_prev * d_v_prev__d_y
                )
                +
                KINEMATIC_VISCOSITY * laplace__v_prev
            )
        )

        # Velocity Boundary Conditions: Homogeneous Dirichlet BC everywhere
        # except for the horizontal velocity at the inlet, which is constant
        u_tent = V_x_BC(u_tent)
        v_tent = V_y_BC(v_tent)
        
        d_p_next__d_x = central_difference_x(p_prev)
        d_p_next__d_y = central_difference_y(p_prev)

        # Correct the velocities
        Fx = 0.0
        Fy = 0.0
        u_next = (
            u_tent
            -
            dt 
            *
            (d_p_next__d_x / DENSITY
            +
            Fx)
        )
        v_next = (
            v_tent
            -
            dt 
            *
            (d_p_next__d_y / DENSITY
            +
            Fy)
            
        )

        # Velocity Boundary Conditions: Homogeneous Dirichlet BC everywhere
        # except for the horizontal velocity at the inlet, which is prescribed
        u_next = V_x_BC(u_next)
        v_next = V_y_BC(v_next)
        
        
        B = POISSON_B_TERM(DENSITY, u_next, v_next, dx, dy)
        
        for _I in range(N_PRESSURE_POISSON_ITERATIONS):
            p_next = poisson_second_order(p_prev, dx, dy, 0, 0, 0, 0, ngc, B)
            err =  (p_next - p_prev).max()
            if err < 1e-6:
                
                break
            p_prev = p_next
        
        
        err_v = np.max(np.abs(v_next - v_prev))
        err_u = np.max(np.abs(u_next - u_prev))
        
        if err_u <= 1e-6 and err_v <= 1e-6:
            print('convergence')
            u_prev = u_next
            v_prev = v_next
            p_prev = p_next
            break
        if err_u>1e3 or err_v>1e3:
            print("Error is large and thus solution may not converge.")
            break
        # Advance in time
        u_prev = u_next
        v_prev = v_next
        p_prev = p_next
    

    
    plt.figure()
    plt.plot(X_object, Y_object,linewidth = 1,color = 'r')
    plt.contourf(X[::2, ::2], Y[::2, ::2], p_next[::2, ::2], cmap="coolwarm")
    plt.colorbar()
    plt.quiver(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2], color="black")
    plt.xlim((0, DOMAIN_SIZE_x))
    plt.ylim((0, DOMAIN_SIZE_y))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.title("Flow Around a Circle")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    
    x_i = X[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object + np.sqrt(dx**2 + dy**2)]
    P_r = p_next[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object + np.sqrt(dx**2 + dy**2)]
    P_Back = P_r[x_i>=Center_Object_x].sum()
    P_Front = P_r[x_i<Center_Object_x].sum()
    p_sum = p_next[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object + np.sqrt(dx**2 + dy**2)].sum()
    #Assume Flow and Object are axisymmetric
    print('Drag = ', round((P_Front - P_Back)*np.pi*2*R_Object/2,2), 'N')