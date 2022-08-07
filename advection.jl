using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets
using Plots

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 7.0

N = 32

dx = (x_max-x_min)/N
dy = (y_max-y_min)/N

islocation(x, y) = x > x_max / 2 - dx && x < x_max / 2 + dx && y > y_max / 10 - dx && y < y_max / 10 + dx
emission(x, y, emisrate) = ifelse(islocation(x, y), emisrate, 0)
@register emission(x, y, emisrate)

emisrate=10.0
k=0.01 # k is reaction rate

@parameters x y t
@variables so2(..) so4(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
# Circular winds.
θ(x,y) = atan(y.-0.5, x.-0.5)
u(x,y) = -sin(θ(x,y))
v(x,y) = cos(θ(x,y))
eq = [
    # Advection and reaction.
    Dt(so2(x,y,t)) ~ -u(x,y)*Dx(so2(x,y,t)) - v(x,y)*Dy(so2(x,y,t)) + emission(x, y, emisrate) - k*so2(x,y,t),
    Dt(so4(x,y,t)) ~ -u(x,y)*Dx(so4(x,y,t)) - v(x,y)*Dy(so4(x,y,t)) + k*so2(x,y,t),
]

domains = [x ∈ Interval(x_min, x_max),
              y ∈ Interval(y_min, y_max),
              t ∈ Interval(t_min, t_max)]

# Periodic BCs
bcs = [so2(x,y,t_min) ~ 0.0,
       so2(x_min,y,t) ~ so2(x_max,y,t),
       so2(x,y_min,t) ~ so2(x,y_max,t),

       so4(x,y,t_min) ~ 0.0,
       so4(x_min,y,t) ~ so4(x_max,y,t),
       so4(x,y_min,t) ~ so4(x,y_max,t),
] 

@named pdesys = PDESystem(eq,bcs,domains,[x,y,t],[so2(x,y,t), so4(x,y,t)])

discretization = MOLFiniteDifference([x=>dx, y=>dy], t, approx_order=2, grid_align=center_align)

# Convert the PDE problem into an ODE problem
println("Discretization:")
@time prob = discretize(pdesys,discretization)

println("Solve:")
@time sol = solve(prob, TRBDF2(), saveat=0.1)

# Plotting
discrete_x = x_min:dx:x_max
discrete_y = y_min:dy:y_max

Nx = floor(Int64, (x_max - x_min) / dx) + 1
Ny = floor(Int64, (y_max - y_min) / dy) + 1

@variables so2[1:Nx,1:Ny](t)
@variables so4[1:Nx,1:Ny](t)

anim = @animate for k in 1:length(sol.t)
    solso2 = reshape([sol[so2[(i-1)*Ny+j]][k] for j in 1:Ny for i in 1:Nx],(Ny,Nx))
    solso4 = reshape([sol[so4[(i-1)*Ny+j]][k] for j in 1:Ny for i in 1:Nx],(Ny,Nx))

    p1 = heatmap(discrete_x, discrete_y, solso2[2:end, 2:end], title="t=$(sol.t[k]); so2")
    p2 = heatmap(discrete_x, discrete_y, solso4[2:end, 2:end], title="t=$(sol.t[k]); so4")
    plot(p1, p2, size=(1000,400))
end
gif(anim, "advection.gif", fps = 8)