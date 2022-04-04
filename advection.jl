using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets, Plots

@parameters x y t
@variables so2(..) so4(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

∇²(u) = Dxx(u) + Dyy(u)

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 11.5

islocation(x, y) = x > x_max / 2 - dx && x < x_max / 2 + dx && y > y_max / 10 - dx && y < y_max / 10 + dx
emission(x, y, emisrate) = ifelse(islocation(x, y), emisrate, 0)
@register emission(x, y, emisrate)
emisrate = 10.0;


u = 1.0
v = -1.0

eq = [
    Dt(so2(x,y,t)) ~ u*Dx(so2(x,y,t)) + v*Dy(so2(x,y,t)) + emission(x, y, emisrate),
    Dt(so4(x,y,t)) ~ u*Dx(so4(x,y,t)) + v*Dy(so4(x,y,t)) + emission(x, y, emisrate),
]

domains = [x ∈ Interval(x_min, x_max),
              y ∈ Interval(y_min, y_max),
              t ∈ Interval(t_min, t_max)]

# Periodic BCs
bcs = [so2(x,y,0) ~ 0.0,
       so2(0,y,t) ~ so2(1,y,t),
       so2(x,0,t) ~ so2(x,1,t),

       so4(x,y,0) ~ 0.0,
       so4(0,y,t) ~ so4(1,y,t),
       so4(x,0,t) ~ so4(x,1,t),
] 

@named pdesys = PDESystem(eq,bcs,domains,[x,y,t],[so2(x,y,t), so4(x,y,t)])

N = 32

dx = 1/N
dy = 1/N

order = 2

discretization = MOLFiniteDifference([x=>dx, y=>dy], t, approx_order=order, grid_align=center_align)

# Convert the PDE problem into an ODE problem
println("Discretization:")
@time prob = discretize(pdesys,discretization)

println("Solve:")
@time sol = solve(prob, TRBDF2(), saveat=0.1)

discrete_x = x_min:dx:x_max
discrete_y = y_min:dy:y_max

Nx = floor(Int64, (x_max - x_min) / dx) + 1
Ny = floor(Int64, (y_max - y_min) / dy) + 1

@variables so2[1:Nx,1:Ny](t)
@variables so4[1:Nx,1:Ny](t)

anim = @animate for k in 1:length(sol.t)
    solso2 = reshape([sol[so2[(i-1)*Ny+j]][k] for i in 1:Nx for j in 1:Ny],(Nx,Ny))
    solso4 = reshape([sol[so4[(i-1)*Ny+j]][k] for i in 1:Nx for j in 1:Ny],(Nx,Ny))
println(k)

    p1 = heatmap(solso2[2:end, 2:end], title="$(sol.t[k]) so2")#, clims=(0,5.0))
    p2 = heatmap(solso4[2:end, 2:end], title="$(sol.t[k]) so4")#, clims=(0,5.0))
    plot(p1, p2)
end
gif(anim, "advection.gif", fps = 8)