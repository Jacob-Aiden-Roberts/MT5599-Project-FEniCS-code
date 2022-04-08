"""
FEniCS program describing the evolution of travelling wave in wave-pinning model
for Rho-GTPase polarization. Code is an altered version of the code used to solve "A system of
advection-diffusion-reaction equations" from the book "Solving PDEs in Python - The Fenics
Tutorial Volume 1" at fenicsproject.org/pub/tutorial/html/._ftut1010.html
  
  u' = D_u*Delta(u) + ((a*u**2)/1+k*u**2)*v - b*u
  v' = D_v*Delta(v) - ((a*u**2)/1+k*u**2)*v + b*u

This file simulates a 2D domain.
"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

#%%

# Define parameters
T = 600 # Final time
num_steps = 300  # Number of time steps
dt = T/num_steps   # Time step size
D_u = 0.01   # Diffusion constant for active species
D_v = 1   # Diffusion constant for inactive species
a = 1   # Parameter a
b = 1   # Parameter b
k = 0.01   # Parameter k

# Create mesh
domain = Circle(Point(1, 1), 1)
mesh = generate_mesh(domain, 64)


P1 = FiniteElement('P', triangle, 1) # Define single finite element unit
element = MixedElement([P1, P1,]) # Apply this to a mixed element to apply to both u and v
V = FunctionSpace(mesh, element) # Define function space using mesh and element

# Define test functions
w_u, w_v = TestFunctions(V)

# Define functions for GTPase
uv = Function(V) # u/v at time n
uv_n = Function(V) # u/v at time n+dt

#Set Initial Conditions

# This creates the travelling wave solution seen in figures 10, 11 and 12 in the report. 
# To create a plot similar to figure 12 simply increase the coefficient of the inactive GTPase (v, the second value) by some amount
#u_0 = Expression(('abs(x[0]*x[1])', 'abs(5*x[1]*x[0])'), degree=1)
#uv = interpolate(u_0, V)


# Change in polarity. 
# This will create figures 13, 14 and 15. The initial conditions creates concentrated peaks in two places on the 
# domain for the  active GTPase, u. While creating a very dispersed (and so approximately homogenous after T=1) peak in
# the centre of the domain for the inactive GTPase, v.
u_0 = Expression(('exp(-c*pow(x[0]-1.5, 2) - c*pow(x[1]-1.5, 2)) + 5*exp(-c*pow(x[0]-0.5, 2) - c*pow(x[1]-0.5, 2))','5*exp(-(1/c)*pow(x[0]-1, 2) - (1/c)*pow(x[1]-1, 2))'),
                 degree=1, c=100)
uv = interpolate(u_0, V)



# Split trial functions to two components
u, v = split(uv)
u_n, v_n = split(uv_n)

# Define constants used in variational forms
a = Constant(a)
b = Constant(b)
k = Constant(k)
D_u = Constant(D_u)
D_v = Constant(D_v)
delta = Constant(dt)

# Define F(u,v). This allows for the solving of different reaction equations with less editing
# Simply comment out the F that you do not wish to use for this run of the code
#F = ( ( a*u_n**2)/(1 + k*u_n**2) )*v_n - b*u_n # Wave-Pinning
F = ( ( a*u_n**2)*v_n - b*u_n ) # Turing-Instability


# Define variational problem
var = ( (u_n - u)/delta )*w_u*dx + D_u*dot(grad(u_n), grad(w_u))*dx \
	+ ( (v_n - v)/delta )*w_v*dx + D_v*dot(grad(v_n), grad(w_v))*dx \
	- F*w_u*dx \
	+ F*w_v*dx

# Create VTK files for output
vtkfile_u = File("Wave_Pinning/u.pvd")
vtkfile_v = File("Wave_Pinning/v.pvd")


# Time-stepping
t = 0
for n in range(num_steps):

	t += dt # Update time
	print("The time is:",t)
	solve(var == 0, uv_n,solver_parameters={"newton_solver":{"relative_tolerance":1e-10},"newton_solver":{"maximum_iterations":10000}}) # Solve variational problem for current time step
	print("Solve was carried out")

	# Save solution to VTK files
	# These are used with 'paraview' in order to visualise the solutions more precisely than is done by matplotlib
	_u_, _v_ = uv.split()
	vtkfile_u << (_u_, t)
	vtkfile_v << (_v_, t)

	uv.assign(uv_n) # Update solution

# Plot solution
plt.figure()
plt.subplot(411)
plot(_u_)
plt.title('Active GTPase')

plt.subplot(412)
plot(_v_)
plt.title('Inactive GTPase')
plt.tight_layout()
#plt.colorbar()
plt.savefig("GTPase.png")	










