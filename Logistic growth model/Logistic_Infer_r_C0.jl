#=
    Logistic_Infer_r_C0.jl
    Produces inference and information geometry results for the logistic model, estimating θ = (r,C(0))
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
include("FIM_Obs_Process.jl")
include("Logistic_partials.jl")
include("Logistic_Jacobians.jl")
include("LogisticFisherInformation.jl")
include("LogisticSolAllParams.jl")
include("DataGenerateLogistic.jl")
include("../ConfidenceRegions.jl")
include("../InfoGeo2D.jl")
include("../optimise.jl")
include("../EquidistantCircumferencePoints.jl")
gr()
default(fontfamily="Arial")

TimePoints = [1000/365.25,2500/365.25,4000/365.25] #timepoints at which to to generate synthetic data
NperT = [10,10,10] #number of observations at each time in TimePoints

#True values used to generate data
σ = 2.301 #standard deviation of synthetic data
r = 0.0025*365.25 #rate of growth
C = 0.7237 #initial coverage (C(0))
K = 79.74  #carrying capacity
θtrue = [r,C]
Ntime,Ndata = DataGenerateLogistic([r,C,K],TimePoints,NperT,σ) #generate synthetic data
#Initial guesses
C0 = 0.2 #initial coverage (C(0))
r0 = 0.003*365.25 #rate of growth
θ0 = [r0,C0]
# Bounds for optimisation to determine MLE
r_min = 0.0 #lower bound on r
r_max = 0.05*365.25 #upper bound on r
C0_min = 0.0 #lower bound on C(0)
C0_max = 1.0 #upper bound on C(0)
lb = [r_min,C0_min] #lower bounds
ub = [r_max,C0_max] #upper bounds

#Solution to the logistic model as a function of the parameters and time
function LogisticSol(θ,t,K)
    (r,C0) = θ
    Logistic = K*C0/(C0+(K-C0)*exp(-r*t))
    return Logistic
end

#Loglikleihood function
llfun = (θ) -> sum([ loglikelihood(Normal(LogisticSol(θ[1:2],t,K),σ),[Ndata[i]]) for (i,t) in enumerate(Ntime)])
#Compute MLE and MLL
MLE,MLL  = optimise(llfun,θ0,lb,ub)

#PLot solution to model where parameters are given by the MLE
Tspan = range(0,Ntime[end],length = 100)
LogisticSolMLE = zeros(length(Tspan))
for (i,t) in enumerate(Tspan)
LogisticSolMLE[i] = LogisticSol(MLE,t,K)
end
pltsol=plot(Tspan,LogisticSolMLE,label="Model fit")
scatter!(pltsol,Ntime, Ndata,label="data",xlabel = "Time (years)",ylabel = "Population density (%)",title = "Logistic infer r C0")
#Add solution based on true parameters to the plot
Logistic_sol=zeros(length(Tspan))
for (i,t) in enumerate(Tspan)
Logistic_sol[i] = LogisticSol([r,C],t,K)
end
plot!(pltsol,(Tspan,Logistic_sol),label = "True model",color = :red, markerstrokewidth = 1.5,legend=:bottomright,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,box = :on, grid = :false)

#Specify range for loglikelihood and scalar curvature surface plots
rrange = range(0.85,1.05,length=100)
C0range = range(0.3,1.0,length=100)

#Compute, normalise and plot loglikelihood surface
LogLikeSurface = [llfun([rval,cval]) for rval in rrange, cval in C0range]
NormalisedLogLikelihood = LogLikeSurface .- MLL
pltLogLikeSurface = heatmap(C0range,rrange,NormalisedLogLikelihood,xlabel = "C₀", ylabel = "r",right_margin = 20Plots.mm,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color = :oslo,clims=(-10,0))
pltLogLikeSurface = scatter!(pltLogLikeSurface,(MLE[2],MLE[1]))

#Compute and plot confidence regions
CR = confidenceregion2D(MLE,MLL,llfun)
CRplot = deepcopy(pltLogLikeSurface)
CRplot = plot!(CRplot,CR[:,2],CR[:,1],label="0.95 CR",linecolor=:magenta)

#Construct Fisher information matrix for the logistic model, estimating θ = (r,C(0))
function Fisher(θ)
    FIM = Fisher_rC([θ;K],σ,TimePoints,NperT)
    return FIM
end

#Compute and plot the scalar curvature surface (scalar curvature is third output of RiemannTensor function)
ScalarSurface = [RiemannTensor([rval,Cval],5,Fisher)[3] for rval in rrange, Cval in C0range]
ScalarSurfaceplot = heatmap(C0range,rrange,ScalarSurface,xlabel = "C₀", ylabel = "r",right_margin = 20Plots.mm,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color=:davos,clims=(-0.005,0))

vrange = EquidistantCircumferencePoints(20) #generate 20 initial velocities for geodesic curves
for k in 1:size(vrange,1) #produce geodesics with different initial velocities (in different directions)
#Geodesics to prescribed Length
global CRplot
global ScalarSurfaceplot
targetLength = sqrt(quantile(Chisq(2),0.95)) #geodesic length coresponding to theoretical 95% confidence
#Obtain geodesic
t,sol  = SingleGeodesic2D(MLE,vrange[k,:],Tsit5(),targetLength,Fisher,cb_single2D)
a = [sol[i][1] for i=1:length(sol)]
b = [sol[i][2] for i=1:length(sol)]
#Plot geodesics onto the loglikelihood and scalar surface plots
CRplot = plot!(CRplot,b,a,legend = false,linecolor=:black)
ScalarSurfaceplot = plot!(ScalarSurfaceplot,b,a,legend = false,linecolor=:black)
end
#Add confidence region, MLE and true parameters to scalar curvature plot
ScalarSurfaceplot = plot!(ScalarSurfaceplot,CR[:,2],CR[:,1],linecolor = :magenta)
ScalarSurfaceplot = scatter!(ScalarSurfaceplot,(θtrue[2],θtrue[1]),markercolor=:green)
ScalarSurfaceplot = scatter!(ScalarSurfaceplot,(MLE[2],MLE[1]),markercolor=:red,framestyle=:box,widen=false)
#Add MLE and true parameters to loglikelihood plot
CRplot = scatter!(CRplot,(θtrue[2],θtrue[1]),markercolor=:green)
CRplot = scatter!(CRplot,(MLE[2],MLE[1]),markercolor=:red,framestyle=:box,widen=false)
