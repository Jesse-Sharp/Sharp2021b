#=
    MvNormalModel_HT.jl
    Performs example hypothesis tests for the multivariate normal distribution estimating θ = (μ₁,μ₂)
=#

#Initialisation (packages)
using Random
using Distributions
using Plots
using LaTeXStrings
using StatsBase
using NLopt
using CSV
using DataFrames
using ForwardDiff
using FiniteDiff
using FiniteDifferences
using DifferentialEquations
using Roots
using .Threads
using LinearAlgebra
#Initialisation (local functions and plot settings)
include("optimise.jl")
include("InfoGeo2D.jl")
include("EquidistantCircumferencePoints.jl")
include("ConfidenceRegions.jl")
gr()
default(fontfamily="Arial")


N = 10 #number of observations
μ = [0.8;1.2] #true means used to generate data
θtrue = μ
σ = 0.3 #known standard deviation used to generate data

θ0 = [1,1] #initial guess
lb = [0,0] #lower bounds
ub = [2,2] #upper bounds

#Set random seed for reproducibility
Random.seed!(4107)

MvN = MvNormal(μ,σ ^ 2 * I) #multivariate normal distributuion with means μ and standard deviation σ
Data = rand(MvN,N) #generate N synthetic datapoints
llfun = θ -> loglikelihood(MvNormal(θ,σ ^ 2 * I),Data) #loglikelihood function
MLE,MLL  = optimise(llfun,θ0,lb,ub) #compute maximum loglikelihood estimate (point-estimate of parameters), and corresponding loglikelihood value

#Specify range for loglikelihood surface plot
μ1range = range(0.4,1.0,length=100)
μ2range = range(0.9,1.5,length=100)

#Compute, normalise and plot loglikelihood surface
LogLikeSurface = [llfun([μ1val,μ2val]) for μ1val in μ1range, μ2val in μ2range]
NormalisedLogLikelihood = LogLikeSurface .- MLL
pltLogLikeSurface = heatmap(μ2range,μ1range,NormalisedLogLikelihood,xlabel = "μ₂", ylabel = "μ₁",right_margin = 20Plots.mm,bottom_margin = 5Plots.mm,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color=:oslo,framestyle = :box,widen=false,clims=(-10,0))
pltLogLikeSurface = scatter!(pltLogLikeSurface,(MLE[2],MLE[1]))

#Compute and plot confidence regions
CR = confidenceregion2D(MLE,MLL,llfun)
CRplot = deepcopy(pltLogLikeSurface)
CRplot = plot!(CRplot,CR[:,2],CR[:,1],label="0.95 CR",linecolor=:magenta)

#Construct Fisher information matrix for the observation process
function Fisher(θ)
    FIM = N .* diagm([1/(σ^2); 1/(σ^2)])
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
#Add MLE and true parameters to loglikelihood plot
CRplot = scatter!(CRplot,(θtrue[2],θtrue[1]),markercolor=:green)
CRplot = scatter!(CRplot,(MLE[2],MLE[1]),markercolor=:red,framestyle=:box,widen=false)

Ntests = 2 #Number of tests to perform
#Parameters to test
θtest = Array{Vector{Float64},1}(undef,Ntests)
θtest[1] = [0.8,1.0]
θtest[2] = [0.9,1.4]
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
