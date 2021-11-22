#=
    Exponential_Infer_C0_StDev.jl
    Produces inference and information geometry results for the exponential model, estimating θ = (C0,σ)
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
θtrue = [C0,σ]
Ntime,Ndata = DataGenerateExponential([a,C0],TimePoints,NperT,σ)
#Initial guesses
C0_0 = 0.3 #initial condition (C(0))
σ0 = 0.3 #standard deviation of synthetic data
θ0 = [C0_0,σ0]
# Bounds for optimisation to determine MLE
C0_min = 0.0 #lower bound on C(0)
C0_max = 1.0 #upper bound on C(0)
σ_min = 0.0 #lower bound on σ
σ_max = 1.0 #upper bound on σ
lb = [C0_min,σ_min] #lower bounds
ub = [C0_max,σ_max] #upper bounds

#Solution to the exponential model as a function of the parameters and time
function ExponentialSol(θ,t)
    (a,C0) = θ
    Exponential = ExponentialSolAllParams([a;C0],t)
    return Exponential
end

#Loglikleihood function
llfun = (θ) -> sum([ loglikelihood(Normal(ExponentialSol([a;θ[1]],t),θ[2]),[Ndata[i]]) for (i,t) in enumerate(Ntime)])
#Compute MLE and MLL
MLE,MLL  = optimise(llfun,θ0,lb,ub)

#PLot solution to model where parameters are given by the MLE
Tspan = range(0,Ntime[end],length = 100)
ExponentialSolMLE = zeros(length(Tspan))
for (i,t) in enumerate(Tspan)
ExponentialSolMLE[i] = ExponentialSol([a,MLE[1]],t)
end
pltsol=plot(Tspan,ExponentialSolMLE,label="Model fit")
scatter!(pltsol,Ntime, Ndata,label="data",xlabel = "time",ylabel = "Density",title = "Logistic")
#Add solution based on true parameters to the plot
Exponential_sol=zeros(length(Tspan))
for (i,t) in enumerate(Tspan)
Exponential_sol[i] = ExponentialSol([a;θtrue[1]],t)
end
plot!(pltsol,(Tspan,Exponential_sol),label = "Underlying model",color = :red, markerstrokewidth = 1.5,legend=:bottomright)

#Specify range for loglikelihood and scalar curvature surface plots
C0range = range(0.55,0.80,length=100)
σrange = range(0.15,0.35,length=100)

#Compute, normalise and plot loglikelihood surface
LogLikeSurface = [llfun([C0val,σval]) for C0val in C0range, σval in σrange]
NormalisedLogLikelihood = LogLikeSurface .- MLL
pltLogLikeSurface = heatmap(σrange,C0range,NormalisedLogLikelihood,xlabel = "σ", ylabel = "C0",right_margin = 20Plots.mm,bottom_margin = 5Plots.mm,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color=:oslo,clims=(-10,0))
pltLogLikeSurface = scatter!(pltLogLikeSurface,(MLE[2],MLE[1]))

#Compute and plot confidence regions
CR = confidenceregion2D(MLE,MLL,llfun)
CRplot = deepcopy(pltLogLikeSurface)
CRplot = plot!(CRplot,CR[:,2],CR[:,1],label="0.95 CR",linecolor = :magenta)

#Construct Fisher information matrix for the exponential model, estimating θ = (C(0),σ)
function Fisher(θ)
    FIM = Fisher_C0_StDev([a;θ[1]],θ[2],TimePoints,NperT)
    return FIM
end

#Compute and plot the scalar curvature surface (scalar curvature is third output of RiemannTensor function)
ScalarSurface = [RiemannTensor([C0val,σval],5,Fisher)[3] for C0val in C0range, σval in σrange]
ScalarSurfaceplot = heatmap(σrange,C0range,ScalarSurface,right_margin = 20Plots.mm,bottom_margin = 5Plots.mm,xlabel = "σ", ylabel = "C0",xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color=:davos,clims=(-0.05,0))

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
