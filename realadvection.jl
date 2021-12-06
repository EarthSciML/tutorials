using OrdinaryDiffEq, ModelingToolkit, DiffEqOperators, DomainSets, Distributions, NetCDF
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
u_exact = (x,t) -> exp.(-t) * cos.(x)
@parameters t x y
@variables u(..)
Dx = Differential(x)
Dy = Differential(y)
#Dxx = Differential(x)^2
#Dyy = Differential(y)^2
Dt = Differential(t)

#= ncinfo("/Users/Minwoo/Desktop/emis/emis_mole_all_20160701_cb6_bench.nc")
ncinfo("/Users/Minwoo/Desktop/emis/METDOT3D_160701.nc")
#101×81×35×25 Array{Float32, 4} /// 35 x 25???  35- layer =#

t_min= 0.
t_max = 24
x_min = 0.
x_max = 12000*101
#may be 101?
y_min = 0.
y_max = 12000*81

#Getting real dat from CMAQ
emisission_file = "/Users/Minwoo/Desktop/emis/METDOT3D_160701.nc"


nx_emis = ncgetatt(emisission_file, "Global", "NCOLS")
ny_emis = ncgetatt(emisission_file, "Global", "NROWS")
nt_emis = 24
dx_emis = ncgetatt(emisission_file, "Global", "XCELL")
dy_emis = ncgetatt(emisission_file, "Global", "YCELL")

#Getting Uhat data
r1 = ncread("/Users/Minwoo/Desktop/emis/METDOT3D_160701.nc","UHAT_JD")


function u_hat(x,y)
    i = Int(ceil((x - x_min)/dx_emis))
    j = Int(ceil((y - y_min)/dy_emis))
    #println(x, " ",y," ", i," ", j)
    u = r1[:, :, 1, 1]'
    return u[j,i]
end

@register u_hat(x, y)

#Getting Vhat data
r2 = ncread("/Users/Minwoo/Desktop/emis/METDOT3D_160701.nc","VHAT_JD")


function v_hat(x,y)
    i = Int(ceil((x - x_min)/dx_emis))
    j = Int(ceil((y - y_min)/dy_emis))
    #println(x, " ",y," ", i," ", j)
    v = r2[:, :, 1, 1]'
    return v[j,i]
end
@register v_hat(x, y)

#So2 Concetration
r3 = ncread("/Users/Minwoo/Desktop/emis/emis_mole_all_20160701_cb6_bench.nc","SO2")
emis = r3[:, :, 1, 1]'

function emissions(x,y)
    i = Int(ceil((x - x_min)/dx_emis))
    j = Int(ceil((y - y_min)/dy_emis))
    #println(x, " ",y," ", i," ", j)
    return emis[j,i]
end


#= function emissions(x,y,t)
    i = Int(ceil((x - x_min)/dx_emis))
    j = Int(ceil((y - y_min)/dy_emis))
    k = maximum([Int(ceil((t - t_min)/dt_emis)), 1])
    #println(x, " ",y," ",t," ", i," ", j," ",k)
    return emis[i,j,k]
end =#

@register emissions(x, y)

#@register emissions(x, y, t)

# 3D PDE

eq  = Dt(u(t,x,y)) ~ u_hat(x,y)*Dx(u(t,x,y)) + v_hat(x,y)*Dy(u(t,x,y)) + emissions(x, y)
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
dx = 24000.0; dy = 24000.0; dt = 1;
#dx = 12000
discretization = MOLFiniteDifference([x=>dx,y=>dy],t)
prob = ModelingToolkit.discretize(pdesys,discretization)
#sol = solve(prob,Tsit5())
#sol = solve(prob,saveat = dt, Tsit5())
sol = solve(prob,saveat = dt, progress = true, progress_steps = 1,Tsit5())

using Plots
_, xs,ys = [infimum(d.domain):dx:supremum(d.domain) for d in domains]


u_sol = reshape(sol.u[1], length(xs)-1,length(ys)-1)
p1 = plot(xs[2:length(xs)], ys[2:length(ys)], u_sol', linetype=:contourf,title = "solution")
u_sol = reshape(sol.u[2], length(xs)-1,length(ys)-1)
p2 = plot(xs[2:length(xs)], ys[2:length(ys)], u_sol', linetype=:contourf,title = "solution")
u_sol = reshape(sol.u[3], length(xs)-1,length(ys)-1)
p3 = plot(xs[2:length(xs)], ys[2:length(ys)], u_sol', linetype=:contourf,title = "solution")
u_sol = reshape(sol.u[4], length(xs)-1,length(ys)-1)
p4 = plot(xs[2:length(xs)], ys[2:length(ys)], u_sol', linetype=:contourf,title = "solution")
plottogether = plot(p1,p2,p3,p4)



julia> p1 = heatmap(xs[2:length(xs)-1], ys[2:length(ys)-1], u_sol', title = "solution")