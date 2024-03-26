# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:48:01 2024

@author: jacob
"""

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


* Velocity and pressure have zero initial condition.
* Homogeneous Dirichlet Boundary Conditions everywhere except for horizontal
  velocity at top. It is driven by an external flow.

-----

Solution strategy:   (Projection Method: Chorin's Splitting)

1. Solve Momentum equation without pressure gradient for tentative velocity
   (with given Boundary Conditions)

    ∂u/∂t + (u ⋅ ∇) u = ν ∇²u

2. Solve pressure poisson equation for pressure at next point in time
   (with homogeneous Neumann Boundary Conditions everywhere except for
   the top, where it is homogeneous Dirichlet)

    ∇²p = ρ/Δt ∇ ⋅ u           

3. Correct the velocities (and again enforce the Velocity Boundary Conditions)

    u ← u − Δt/ρ ∇ p

-----

Strategy in index notation

u = [u, v]
x = [x, y]

1. Solve tentative velocity + velocity BC

    ∂u/∂t + u ∂u/∂x + v ∂u/∂y = ν ∂²u/∂x² + ν ∂²u/∂y²

    ∂v/∂t + u ∂v/∂x + v ∂v/∂y = ν ∂²v/∂x² + ν ∂²v/∂y²

2. Solve pressure poisson + pressure BC

    ∂²p/∂x² + ∂²p/∂y² = ρ/Δt (∂u/∂x + ∂v/∂y)

3. Correct velocity + velocity BC

    u ← u − Δt/ρ ∂p/∂x

    v ← v − Δt/ρ ∂p/∂y

------

IMPORTANT: Take care to select a timestep that ensures stability
"""

import matplotlib.pyplot as plt
import numpy as np

N_POINTS = 152
DOMAIN_SIZE = 1.0
N_ITERATIONS = 100
TIME_STEP_LENGTH = 0.0001
KINEMATIC_VISCOSITY = 0.01
DENSITY = 1.0
HORIZONTAL_VELOCITY_TOP = 1.0

DYNAMIC_VISCOSITY = KINEMATIC_VISCOSITY/DENSITY
RE = DENSITY*HORIZONTAL_VELOCITY_TOP*DOMAIN_SIZE/DYNAMIC_VISCOSITY

N_PRESSURE_POISSON_ITERATIONS = 500
STABILITY_SAFETY_FACTOR = 0.5


########################################################################
#Object Inputs
Object_Boolean = True
if Object_Boolean:
    R_Object = 0.15
    Center_Object_x = 0.5
    Center_Object_y = 0.5
########################################################################

if __name__ == "__main__":
    element_length = DOMAIN_SIZE / (N_POINTS - 1)
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)

    X, Y = np.meshgrid(x, y)
    
    if Object_Boolean:
        X_object = np.round(R_Object*np.cos(np.linspace(0, 2*np.pi,75)) + Center_Object_x,8)
        Y_object = np.round(R_Object*np.sin(np.linspace(0, 2*np.pi,75)) + Center_Object_y,8)

    u_prev = np.zeros_like(X)
    v_prev = np.zeros_like(X)
    p_prev = np.zeros_like(X)

    def Circle_BC(vector, X, Y, Center_Object_x, Center_Object_y, R_Object):
        vector[np.sqrt( (X-Center_Object_x)**2 + (Y-Center_Object_y)**2 ) <= R_Object] = 0
        return vector

    def central_difference_x(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 2:  ]
            -
            f[1:-1, 0:-2]
        ) / (
            2 * element_length
        )
        return diff
    
    def central_difference_y(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[2:  , 1:-1]
            -
            f[0:-2, 1:-1]
        ) / (
            2 * element_length
        )
        return diff
    
    def laplace(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 0:-2]
            +
            f[0:-2, 1:-1]
            -
            4
            *
            f[1:-1, 1:-1]
            +
            f[1:-1, 2:  ]
            +
            f[2:  , 1:-1]
        ) / (
            element_length**2
        )
        return diff
    

    maximum_possible_time_step_length = (
        0.5 * element_length**2 / KINEMATIC_VISCOSITY
    )
    if TIME_STEP_LENGTH > STABILITY_SAFETY_FACTOR * maximum_possible_time_step_length:
        raise RuntimeError("Stability is not guarenteed")

    
    for _ in range(N_ITERATIONS):
        print(_)
        d_u_prev__d_x = central_difference_x(u_prev)
        d_u_prev__d_y = central_difference_y(u_prev)
        d_v_prev__d_x = central_difference_x(v_prev)
        d_v_prev__d_y = central_difference_y(v_prev)
        laplace__u_prev = laplace(u_prev)
        laplace__v_prev = laplace(v_prev)

        # Perform a tentative step by solving the momentum equation without the
        # pressure gradient
        u_tent = (
            u_prev
            +
            TIME_STEP_LENGTH * (
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
            TIME_STEP_LENGTH * (
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
        # except for the horizontal velocity at the top, which is prescribed
        u_tent[0, :] = 0.0 #Bottom
        u_tent[:, 0] = HORIZONTAL_VELOCITY_TOP #Left
        u_tent[:, -1] = u_tent[:, -2] #Right
        u_tent[-1, :] = 0.0 #top
        v_tent[0, :] = 0.0 #Bottom
        v_tent[:, 0] = 0.0 #Left
        v_tent[:, -1] = v_tent[:, -2] #Right
        v_tent[-1, :] = 0.0 #Top
        
        u_tent = Circle_BC(u_tent, X, Y, Center_Object_x, Center_Object_y, R_Object)
        v_tent = Circle_BC(v_tent, X, Y, Center_Object_x, Center_Object_y, R_Object)
        
        d_u_tent__d_x = central_difference_x(u_tent)
        d_v_tent__d_y = central_difference_y(v_tent)

        # Compute a pressure correction by solving the pressure-poisson equation
        rhs = (
            DENSITY / TIME_STEP_LENGTH
            *
            (
                d_u_tent__d_x
                +
                d_v_tent__d_y
            )
        )

        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_next = np.zeros_like(p_prev)
            p_next[1:-1, 1:-1] = 1/4 * (
                +
                p_prev[1:-1, 0:-2]
                +
                p_prev[0:-2, 1:-1]
                +
                p_prev[1:-1, 2:  ]
                +
                p_prev[2:  , 1:-1]
                -
                element_length**2
                *
                rhs[1:-1, 1:-1]
            )

            # Pressure Boundary Conditions: Homogeneous Neumann Boundary
            # Conditions everywhere except for the top, where it is a
            # homogeneous Dirichlet BC
            p_next[:, -1] = p_next[:, -2] #Right
            p_next[0,  :] = p_next[1,  :] #Bottom
            p_next[:,  0] = p_next[:,  1] #Left
            p_next[-1, :] = p_next[-2, :] #Top

            p_prev = p_next
        

        d_p_next__d_x = central_difference_x(p_next)
        d_p_next__d_y = central_difference_y(p_next)

        # Correct the velocities such that the fluid stays incompressible
        u_next = (
            u_tent
            -
            TIME_STEP_LENGTH / DENSITY
            *
            d_p_next__d_x
        )
        v_next = (
            v_tent
            -
            TIME_STEP_LENGTH / DENSITY
            *
            d_p_next__d_y
        )

        # Velocity Boundary Conditions: Homogeneous Dirichlet BC everywhere
        # except for the horizontal velocity at the top, which is prescribed
        u_next[0, :] = 0.0 #Bottom
        u_next[:, 0] = HORIZONTAL_VELOCITY_TOP #Left
        u_next[:, -1] = u_next[:, -2] #Right
        u_next[-1, :] = 0.0 #Top
        v_next[0, :] = 0.0 #Bottom
        v_next[:, 0] = 0.0 #Left
        v_next[:, -1] = v_next[:, -2] #Right
        v_next[-1, :] = 0.0 #top

        u_next = Circle_BC(u_next, X, Y, Center_Object_x, Center_Object_y, R_Object)
        v_next = Circle_BC(v_next, X, Y, Center_Object_x, Center_Object_y, R_Object)
        
        # Advance in time
        u_prev = u_next
        v_prev = v_next
        p_prev = p_next
    

    # The [::2, ::2] selects only every second entry (less cluttering plot)
    #plt.style.use("dark_background")
    plt.figure()
    plt.plot(X_object, Y_object,linewidth = 1,color = 'r')
    plt.contourf(X[::2, ::2], Y[::2, ::2], p_next[::2, ::2], cmap="coolwarm")
    plt.colorbar()
    #plt.quiver(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2], color="black")
    plt.streamplot(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2], color="black")
    plt.xlim((0, DOMAIN_SIZE))
    plt.ylim((0, DOMAIN_SIZE))
    plt.show()

