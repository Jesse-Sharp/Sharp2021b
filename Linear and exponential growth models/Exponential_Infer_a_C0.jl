#=
    Exponential_Infer_a_C0.jl
    Produces inference and information geometry results for the exponential model, estimating θ = (a,C(0))
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
include("Exponential_partials.jl")
include("Exponential_Jacobians.jl")
include("ExponentialFisherInformation.jl")
include("../InfoGeo2D.jl")
include("ExponentialSolAllParams.jl")
include("DataGenerateExponential.jl")
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
θtrue = [a,C0]
Ntime,Ndata = DataGenerateExponential(θtrue,TimePoints,NperT,σ)
#Initial guesses
a0 = 0.003*365.25 #rate of growth
C0_0 =  0.2 #initial condition (C(0))
θ0 = [a0,C0_0]
# Bounds for optimisation to determine MLE
a_min = 0.0 #lower bound on a
a_max = 0.05*365.25 #upper bound on a
C0_min = 0.0 #lower bound on C(0)
C0_max = 10.0 #upper bound on C(0)
lb = [a_min,C0_min] #lower bounds
ub = [a_max,C0_max] #upper bounds

#Solution to the exponential model as a function of the parameters and time
function ExponentialSol(θ,t)
    Exponential = ExponentialSolAllParams(θ,t)
    return Exponential
end

#Loglikleihood function
llfun = (θ) -> sum([ loglikelihood(Normal(ExponentialSol(θ[1:2],t),σ),[Ndata[i]]) for (i,t) in enumerate(Ntime)])
#Compute MLE and MLL
MLE,MLL  = optimise(llfun,θ0,lb,ub)

#PLot solution to model where parameters are given by the MLE
Tspan = range(0,Ntime[end],length = 100)
ExponentialSolMLE = zeros(length(Tspan))
for (i,t) in enumerate(Tspan)
ExponentialSolMLE[i] = ExponentialSol(MLE,t)
end
pltsol=plot(Tspan,ExponentialSolMLE,label="Model fit")
scatter!(pltsol,Ntime, Ndata,label="data",xlabel = "time",ylabel = "Population density (%)",title = "Exponential")
#Add solution based on true parameters to the plot
Exponential_sol=zeros(length(Tspan))
for (i,t) in enumerate(Tspan)
Exponential_sol[i] = ExponentialSol(θtrue,t)
end
plot!(pltsol,(Tspan,Exponential_sol),label = "Underlying model",color = :red, markerstrokewidth = 1.5,legend=:bottomright,ylims = (0.0,1.5))

#Specify range for loglikelihood and scalar curvature surface plots
arange = range(0.25,1.75,length=100)
C0range = range(0.45,0.90,length=100)

#Compute, normalise and plot loglikelihood surface
LogLikeSurface = [llfun([aval,Cval]) for aval in arange, Cval in C0range]
NormalisedLogLikelihood = LogLikeSurface .- MLL
pltLogLikeSurface = heatmap(C0range,arange,NormalisedLogLikelihood,xlabel = "C0", ylabel = "a",right_margin = 20Plots.mm,bottom_margin = 5Plots.mm,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color=:oslo,clims=(-10,0))
pltLogLikeSurface = scatter!(pltLogLikeSurface,(MLE[2],MLE[1]))

#Compute and plot confidence regions
CR = confidenceregion2D(MLE,MLL,llfun)
CRplot = deepcopy(pltLogLikeSurface)
CRplot = plot!(CRplot,CR[:,2],CR[:,1],label="0.95 CR",linecolor=:magenta)

#Construct Fisher information matrix for the exponential model, estimating θ = (a,C(0))
function Fisher(θ)
    FIM = Fisher_aC0(θ,σ,TimePoints,NperT)
    return FIM
end

#Compute and plot the scalar curvature surface (scalar curvature is third output of RiemannTensor function)
ScalarSurface = [RiemannTensor([aval,C0val],5,Fisher)[3] for aval in arange, C0val in C0range]
ScalarSurfaceplot = heatmap(C0range,arange,ScalarSurface,right_margin = 20Plots.mm,bottom_margin = 5Plots.mm,xlabel = "C0", ylabel = "a",xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color=:davos,clims=(-0.05,0))

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
