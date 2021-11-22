#=
    Linear_Infer_a_StDev.jl
    Produces inference and information geometry results for the linear model, estimating θ = (a,σ)
=#

#Initialisation (packages)
using Random
using Distributions
using Plots
using StatsBase
using NLopt
using CSV
using DataFrames
using ForwardDiff
using FiniteDiff
using FiniteDifferences
using DifferentialEquations
import Roots.find_zero
using .Threads
using LinearAlgebra
#Initialisation (local functions and plot settings)
include("../optimise.jl")
include("FIM_Obs_Process.jl")
include("Linear_partials.jl")
include("Linear_Jacobians.jl")
include("LinearFisherInformation.jl")
include("../InfoGeo2D.jl")
include("LinearSolAllParams.jl")
include("DataGenerateLinear.jl")
include("../ConfidenceRegions.jl")
include("../EquidistantCircumferencePoints.jl")
gr()
default(fontfamily="Arial")

TimePoints = [0.1,0.25,0.5] #timepoints at which to to generate synthetic data
NperT = [10,10,10] #number of observations at each time in TimePoints

#True values used to generate data
σ = 2.301/10 #standard deviation of synthetic data
a = 0.0025*365.25 #rate of growth
C0 = 0.7237 #initial condition (C(0))
θtrue = [a,σ]
Ntime,Ndata = DataGenerateLinear([a,C0],TimePoints,NperT,σ)
#Initial guesses
a0 = 0.7 #rate of growth
σ0 = 0.3 #standard deviation of synthetic data
θ0 = [a0,σ0]
# Bounds for optimisation to determine MLE
a_min = 0.0 #lower bound on a
a_max = 1.0 #upper bound on a
σ_min = 0.0 #lower bound on σ
σ_max = 1.0 #upper bound on σ
lb = [a_min,σ_min] #lower bounds
ub = [a_max,σ_max] #upper bounds

#Solution to the linear model as a function of the parameters and time
function LinearSol(θ,t)
    Linear = LinearSolAllParams([θ[1];C0],t)
    return Linear
end

#Loglikleihood function
llfun = (θ) -> sum([ loglikelihood(Normal(LinearSol([θ[1];C0],t),θ[2]),[Ndata[i]]) for (i,t) in enumerate(Ntime)])
#Compute MLE and MLL
MLE,MLL  = optimise(llfun,θ0,lb,ub)

#PLot solution to model where parameters are given by the MLE
Tspan = range(0,Ntime[end],length = 100)
LinearSolMLE = zeros(length(Tspan))
for (i,t) in enumerate(Tspan)
LinearSolMLE[i] = LinearSol(MLE,t)
end
pltsol=plot(Tspan,LinearSolMLE,label="Model fit")
scatter!(pltsol,Ntime, Ndata,label="data",xlabel = "time",ylabel = "Density",title = "Logistic")
#Add solution based on true parameters to the plot
Linear_sol=zeros(length(Tspan))
for (i,t) in enumerate(Tspan)
Linear_sol[i] = LinearSol(θtrue,t)
end
plot!(pltsol,(Tspan,Linear_sol),label = "Underlying model",color = :red, markerstrokewidth = 1.5,legend=:bottomright)

#Specify range for loglikelihood and scalar curvature surface plots
arange = range(0.4,1.1,length=100)
σrange = range(0.15,0.35,length=100)

#Compute, normalise and plot loglikelihood surface
LogLikeSurface = [llfun([aval,σval]) for aval in arange, σval in σrange]
NormalisedLogLikelihood = LogLikeSurface .- MLL
pltLogLikeSurface = heatmap(σrange,arange,NormalisedLogLikelihood,xlabel = "σ", ylabel = "a",right_margin = 20Plots.mm,bottom_margin = 5Plots.mm,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color=:oslo,clims=(-10,0))
pltLogLikeSurface = scatter!(pltLogLikeSurface,(MLE[2],MLE[1]))

CR = confidenceregion2D(MLE,MLL,llfun)
CRplot = deepcopy(pltLogLikeSurface)
CRplot = plot!(CRplot,CR[:,2],CR[:,1],label="0.95 CR",linecolor = :magenta)

#Construct Fisher information matrix for the linear model, estimating θ = (a,σ)
function Fisher(θ)
    FIM = Fisher_a_StDev([θ[1];C0],θ[2],TimePoints,NperT)
    return FIM
end

#Compute and plot the scalar curvature surface (scalar curvature is third output of RiemannTensor function)
ScalarSurface = [RiemannTensor([aval,σval],5,Fisher)[3] for aval in arange, σval in σrange]
ScalarSurfaceplot = heatmap(σrange,arange,ScalarSurface,right_margin = 20Plots.mm,bottom_margin = 5Plots.mm,xlabel = "σ", ylabel = "a",xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color=:davos,clims=(-0.05,0))

vrange = EquidistantCircumferencePoints(20) #generate 20 initial velocities for geodesic curves
for k in 1:size(vrange,1) #produce geodesics with different initial velocities (in different directions)
#Geodesics to prescribed Length
global CRplot
global ScalarSurfaceplot
targetLength = sqrt(quantile(Chisq(2),0.95)) #geodesic length coresponding to theoretical 95% confidence
#Obtain geodesic
t,sol  = SingleGeodesic2D(MLE,vrange[k,:],Tsit5(),targetLength,Fisher,cb_single2D)
c = [sol[i][1] for i=1:length(sol)]
d = [sol[i][2] for i=1:length(sol)]
#Plot geodesics onto the loglikelihood and scalar surface plots
CRplot = plot!(CRplot,d,c,legend = false,linecolor=:black)
ScalarSurfaceplot = plot!(ScalarSurfaceplot,d,c,legend = false,linecolor=:black)
end
#Add confidence region, MLE and true parameters to scalar curvature plot
ScalarSurfaceplot = plot!(ScalarSurfaceplot,CR[:,2],CR[:,1],linecolor = :magenta)
ScalarSurfaceplot = scatter!(ScalarSurfaceplot,(θtrue[2],θtrue[1]),markercolor=:green)
ScalarSurfaceplot = scatter!(ScalarSurfaceplot,(MLE[2],MLE[1]),markercolor=:red,framestyle=:box,widen=false)
#Add MLE and true parameters to loglikelihood plot
CRplot = scatter!(CRplot,(θtrue[2],θtrue[1]),markercolor=:green)
CRplot = scatter!(CRplot,(MLE[2],MLE[1]),markercolor=:red,framestyle=:box,widen=false)
