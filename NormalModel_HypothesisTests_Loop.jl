#=
    NormalModel_HypothesisTests_Loop.jl
    Performs many hypothesis tests for the univariate normal distribution estimating θ = (μ,σ)
=#

#Initialisation (packages)
using Random
using Distributions
using Plots
using StatsBase
using StatsPlots
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
μ = 0.7  #true mean used to generate data
σ = 0.5 #true standard deviation used to generate data
θtrue = [μ,σ]

θ0 = [1,1] #initial guess
lb = [0.01,0.01] #lower bounds
ub = [2,2] #upper bounds

#Set random seed for reproducibility
Random.seed!(4107)

NormDist =  Normal(μ,σ) #normal distributuion with mean μ and standard deviation σ

NumTests = 1000 #Number of datasets to generate and perform hypothesis tests on
λ_LR = Array{Float64,1}(undef,NumTests) #Store test statistics for likelihood-ratio-based hypothesis test
λ_GD = Array{Float64,1}(undef,NumTests) #Store test statistics for geodesic-distance-based hypothesis test

#Construct Fisher information matrix for the observation process
function Fisher(θ)
    FIM = N .* diagm([1/(θ[2]^2); 2/(θ[2]^2)])
    return FIM
end

#Perform hypothesis tests
for test_iter = 1:NumTests
Data = rand(NormDist,N) #generate N synthetic datapoints
llfun = θ -> loglikelihood(Normal(θ[1],θ[2]),Data) #loglikelihood function
MLE,MLL  = optimise(llfun,θ0,lb,ub) #compute maximum loglikelihood estimate (point-estimate of parameters), and corresponding loglikelihood value
λ_LR[test_iter] = -2*(llfun(θtrue)-MLL) #Likelihood-ratio test statistic
Geodesicsol = GeodesicBVPShooting2D(MLE,θtrue,θtrue-MLE,Fisher) #solve BVP for geodesic between MLE and test point
λ_GD[test_iter] = Geodesicsol.u[end][end].^2
end

#Compute what proportion of confidence regions contained the true values
InLRCR = count(λ_LR.<quantile(Chisq(2),0.95))/NumTests #For likelihood-ratio-based confidence regions
InGDCR = count(λ_GD.<quantile(Chisq(2),0.95))/NumTests #For geodesic-distance-based confidence regions

#Plot step histograms of test statistic densities and chisq(2)
plt_hist = plot([λ_LR,λ_GD],seriestype=:stephist,linewidth = 3,linestyle= [:dot :dash], normalize=:pdf, fillcolor=[:blue :orange],fillalpha = 0.25,bins=0:0.5:22,label=["λLR" "λGD"])
plt_hist = plot!(plt_hist,Chisq(2),linecolor=:magenta,linewidth = 2,label="χ(2)",legendfontsize=16,framestyle=:box,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16)
