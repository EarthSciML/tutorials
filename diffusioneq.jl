using OrdinaryDiffEq, ModelingToolkit, DiffEqOperators, DomainSets, Distributions, NetCDF

@parameters t x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)
t_min= 0.
t_max = 24
x_min = 0.
x_max = 12000*100
y_min = 0.
y_max = 12000*80

emisission_file = "/Users/Minwoo/Desktop/emis/emis_mole_all_20160701_cb6_bench.nc"

nx_emis = ncgetatt(emisission_file, "Global", "NCOLS")
ny_emis = ncgetatt(emisission_file, "Global", "NROWS")
nt_emis = 24
dx_emis = ncgetatt(emisission_file, "Global", "XCELL")
dy_emis = ncgetatt(emisission_file, "Global", "YCELL")
#dt_emis = (t_max - t_min)/nt_emis
#add t variable later

#Created random emission, later to be changed by real emission data.

r = ncread("/Users/Minwoo/Desktop/emis/emis_mole_all_20160701_cb6_bench.nc","SO2")
emis = r[:, :, 1, 1]'

#Locate/ align emis into coordinate we're using
function emissions(x,y)
    i = Int(ceil((x - x_min)/dx_emis))
    j = Int(ceil((y - y_min)/dy_emis))
    println(x, " ",y," ", i," ", j)
    return emis[j,i]
end

@register emissions(x, y)

# 3D PDE (diffusion eq + emissions)
D = 1000
eq  = Dt(u(t,x,y)) ~ D * Dxx(u(t,x,y)) + D * Dyy(u(t,x,y)) + emissions(x, y)



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
dx = 12000.; dy = 12000.
discretization = MOLFiniteDifference([x=>dx,y=>dy],t)
prob = ModelingToolkit.discretize(pdesys,discretization)
sol = solve(prob,saveat = 1, Tsit5())

# Plotting
using Plots
ts,xs,ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains]
#= for i ∈ length(sol.t)
    u_sol = reshape(sol.u[i], length(xs)-2,length(ys)-2)
    p = plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")

end
plottogether = plot(p) =#

#Ploting 1 ~ 4 index of the solution
u_sol = reshape(sol.u[1], length(xs)-2,length(ys)-2)
p1 = plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")
u_sol = reshape(sol.u[2], length(xs)-2,length(ys)-2)
p2 = plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")
u_sol = reshape(sol.u[3], length(xs)-2,length(ys)-2)
p3 = plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")
u_sol = reshape(sol.u[4], length(xs)-2,length(ys)-2)
p4 = plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")
plottogether = plot(p1,p2,p3,p4)

u_sol = reshape(sol.u[10], length(xs)-2,length(ys)-2)
p10 = plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol', linetype=:contourf,title = "solution")

#Animation (didn't work)
anim = @animate for i ∈ 1:length(sol.t)
    u_sol = reshape(sol.u[i], length(xs)-2,length(ys)-2)
    plot(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol, linetype=:contourf,title = "solution")


end
gif(anim, "anim_fps15.gif", fps = 15)

