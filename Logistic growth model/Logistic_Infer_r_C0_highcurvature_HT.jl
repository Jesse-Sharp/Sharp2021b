#=
    Logistic_Infer_r_C0_highcurvature_HT.jl
    Performs example hypothesis tests for the logistic model, estimating θ = (r,C(0)), in the high curvature region of parameter space
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
r = 0.9 #rate of growth
C = 0.2 #initial coverage (C(0))
K = 79.74 #carrying capacity
θtrue = [r,C]
Ntime,Ndata = DataGenerateLogistic([r,C,K],TimePoints,NperT,σ)
#Initial guesses
C0 = 0.3 #initial coverage (C(0))
r0 = 1.0  #rate of growth
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

#Specify range for loglikelihood and scalar curvature surface plots
rrange = range(0.75,1.25,length=100)
C0range = range(0.01,0.45,length=400)

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

vrange = EquidistantCircumferencePoints(20) #generate 20 initial velocities for geodesic curves
for k in 1:size(vrange,1) #produce geodesics with different initial velocities (in different directions)
#Geodesics to prescribed Length
global CRplot
targetLength = sqrt(quantile(Chisq(2),0.95)) #geodesic length coresponding to theoretical 95% confidence
#Obtain geodesic
t,sol  = SingleGeodesic2D(MLE,vrange[k,:],Tsit5(),targetLength,Fisher,cb_single2D)
a = [sol[i][1] for i=1:length(sol)]
b = [sol[i][2] for i=1:length(sol)]
#Plot geodesics onto the loglikelihood and scalar surface plots
CRplot = plot!(CRplot,b,a,legend = false,linecolor=:black)
end
#Add confidence region, MLE and true parameters to scalar curvature plot
#Add MLE and true parameters to loglikelihood plot
CRplot = scatter!(CRplot,(θtrue[2],θtrue[1]),markercolor=:green)
CRplot = scatter!(CRplot,(MLE[2],MLE[1]),markercolor=:red,framestyle=:box,widen=false)

Ntests = 4 #Number of tests to perform
#Parameters to test
θtest = Array{Vector{Float64},1}(undef,Ntests)
θtest[1] = [1.0,0.095]
θtest[2] = [0.87,0.25]
θtest[3] = [0.92,0.21]
θtest[4] = [0.9,0.15]
λ_LR = Array{Float64,1}(undef,Ntests) #Store test statistics for likelihood-ratio-based hypothesis test
p_LR = Array{Float64,1}(undef,Ntests) #Store p-value for likelihood-ratio-based hypothesis test
BVP_sol = Array{Any,1}(undef,Ntests) #Store solutions of the geodesic BVPs
λ_GD = Array{Float64,1}(undef,Ntests) #Store test statistics for geodesic-distance-based hypothesis test
p_GD = Array{Float64,1}(undef,Ntests) #Store p-value for geodesic-distance-based hypothesis test

#Perform hypothesis tests
for test_iter = 1:Ntests
#Likelihood-ratio-based hypothesis tests
λ_LR[test_iter] = -2*(llfun(θtest[test_iter])-MLL) #Compute likelihood-ratio-based test statistic
p_LR[test_iter] = 1-cdf(Chisq(2),λ_LR[test_iter]) #Compute likelihood-ratio-based p-value
#Geodesic-distace-based hypothesis tests
BVP_sol[test_iter] = GeodesicBVPShooting2D(MLE,θtest[test_iter],θtest[test_iter]-MLE,Fisher)
λ_GD[test_iter] = BVP_sol[test_iter].u[end][end].^2 #Compute geodesic-distance-based test statistic
p_GD[test_iter] = 1-cdf(Chisq(2),λ_GD[test_iter]) #Compute geodesic-distance-based p-value
#Plotting
global CRplot
CRplot = scatter!(CRplot,(θtest[test_iter][2],θtest[test_iter][1]),markersize = 7,markerstrokewidth=0)
CRplot = plot!(CRplot,BVP_sol[test_iter](range(0,BVP_sol[test_iter].t[end],length = 100))'[:,2],BVP_sol[test_iter](range(0,BVP_sol[test_iter].t[end],length = 100))'[:,1],label = "",linewidth = 2,linecolor = :red)
end

display(CRplot)
