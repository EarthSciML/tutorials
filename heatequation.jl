using OrdinaryDiffEq, ModelingToolkit, DiffEqOperators, DomainSets
# Method of Manufactured Solutions: exact solution
u_exact = (x,t) -> exp.(-t) * cos.(x)

@parameters t x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)
t_min= 0.
t_max = 2.0
x_min = 0.
x_max = 2.
y_min = 0.
y_max = 2.

#xmax = 50.0
#ymax = 30.0
#ncells = 50.0
#dx = x_max / ncells


lecturer(x, y) = x > x_max / 2 - dx && x < x_max / 2 + dx && y > y_max / 10 - dx && y < y_max / 10 + dx
emission(x, y, emisrate) = lecturer(x, y) ? emisrate : 0
@register emission(x, y, emisrate)
emisrate = 10;



# 3D PDE
eq  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y)) + emission(x, y, emisrate)
#emission(x, y, emisrate) = emisrate

analytic_sol_func(t,x,y) = exp(x+y)*cos(x+y+4t)
# Initial and boundary conditionn

bcs = #[u(t_min,x,y) ~ analytic_sol_func(t_min,x,y),
      [u(t_min,x,y) ~ 0,
       u(t,x_min,y) ~ 0,
       u(t,x_max,y) ~ 0,
       u(t,x,y_min) ~ 0,
       u(t,x,y_max) ~ 0]

# Space and time domains
domains = [t ∈ IntervalDomain(t_min,t_max),
           x ∈ IntervalDomain(x_min,x_max),
           y ∈ IntervalDomain(y_min,y_max)]
pdesys = PDESystem([eq],bcs,domains,[t,x,y],[u(t,x,y)])

# Method of lines discretization
dx = 0.1; dy = 0.1
discretization = MOLFiniteDifference([x=>dx,y=>dy],t)
prob = ModelingToolkit.discretize(pdesys,discretization)
sol = solve(prob,Tsit5())

# Ploting
using Plots
xs,ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
u_sol = reshape(sol.u[300], length(xs)-2,length(ys)-2)

plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")

#Animation
anim = @animate for i ∈ 1:length(sol.t)
    u_sol = reshape(sol.u[i], length(xs)-2,length(ys)-2)
    plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")


end
gif(anim, "anim_fps15.gif", fps = 15)

