using ModelingToolkit, MethodOfLines, OrdinaryDiffEq, DomainSets, IfElse
using Plots

@parameters x y t
k = 0.01 # k is reaction rate
@variables so4(..)
@variables xx(..) yy(..) so2_puff(..)
Dt = Differential(t)
Dx = Differential(x)
Dy = Differential(y)

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 7.0

N = 2

dx = (x_max - x_min) / N
dy = (y_max - y_min) / N

# Interaction between Eulerian and Lagrangian frames of reference
puff_conc(x, y, t) = IfElse.ifelse(x - dx / 2 < xx(t),
    IfElse.ifelse(xx(t) < x + dx / 2,
        IfElse.ifelse(y - dy / 2 < yy(t),
            IfElse.ifelse(yy(t) < y + dy / 2, so2_puff(t), 0.0), 0.0), 0.0), 0.0)

puff_conc(x, y, t) = so4(x, y, t)
#@register puff_conc(x, y, t)

# Circular winds.
θ(x, y) = atan(y - 0.5, x - 0.5)
u(x, y) = -sin(θ(x, y))
v(x, y) = cos(θ(x, y))


eq = [
    # Lagrangian puff model.
    Dt(xx(t)) ~ u(xx(t), yy(t)),
    Dt(yy(t)) ~ v(xx(t), yy(t)),
    Dt(so2_puff(t)) ~ -k * so2_puff(t),

    # Eulerian model.
    Dt(so4(x, y, t)) ~ -u(x, y) * Dx(so4(x, y, t)) - v(x, y) * Dy(so4(x, y, t)) + k * so2_puff(t), #puff_conc(x, y, t),
]

domains = [x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max),
    t ∈ Interval(t_min, t_max)]

# Periodic BCs
bcs = [so4(x, y, t_min) ~ 0.0,
    so4(x_min, y, t) ~ so4(x_max, y, t),
    so4(x, y_min, t) ~ so4(x, y_max, t), xx(t_min) ~ 0.5,
    yy(t_min) ~ 0.1,
    so2_puff(t_min) ~ 1.0,
]

@named pdesys = PDESystem(eq, bcs, domains, [x, y, t], [so4(x, y, t), xx(t), yy(t), so2_puff(t)])

discretization = MOLFiniteDifference([x => dx, y => dy], t, approx_order=2, grid_align=center_align)
@time prob = discretize(pdesys, discretization)

println("Solve:")
@time sol = solve(prob, TRBDF2(), saveat=0.1)

# Plotting
discrete_x = x_min:dx:x_max
discrete_y = y_min:dy:y_max

Nx = floor(Int64, (x_max - x_min) / dx) + 1
Ny = floor(Int64, (y_max - y_min) / dy) + 1

@variables so4[1:Nx, 1:Ny](t)
@variables xx(t) yy(t) so2_puff(t)

cmap = cgrad(:inferno);
minval = min(minimum(sol[so2_puff]), minimum([minimum(sol[so4][j][i]) for i ∈ 1:Nx for j ∈ 1:Ny]))
maxval = max(maximum(sol[so2_puff]), maximum([maximum(sol[so4][j][i]) for i ∈ 1:Nx for j ∈ 1:Ny]))
getcolor(v) = get(cmap, (v - minval) / (maxval - minval))

anim = @animate for k in 1:length(sol.t)
    #solso2 = reshape([sol[so2[(i-1)*Ny+j]][k] for j in 1:Ny for i in 1:Nx],(Ny,Nx))
    solso4 = reshape([sol[so4[(i-1)*Ny+j]][k] for j in 1:Ny for i in 1:Nx], (Ny, Nx))

    p1 = scatter([sol[xx][k]], [sol[yy][k]], markercolor=getcolor(sol[so2_puff][k]), legend=:none, xlims=(x_min, x_max), ylims=(y_min, y_max))
    #p1 = heatmap(discrete_x, discrete_y, solso2[2:end, 2:end], title="t=$(sol.t[k]); so2")
    p2 = heatmap(discrete_x, discrete_y, solso4[2:end, 2:end], xlims=(x_min, x_max), ylims=(y_min, y_max), title="t=$(sol.t[k]); so4")
    plot(p1, p2, size=(1000, 400))
end
gif(anim, "advection.gif", fps=8)