#=
    NormalModel.jl
    Produces inference and information geometry results for the univariate normal distribution estimating θ = (μ,σ)
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
μ = 0.7  #true mean used to generate data
σ = 0.5 #true standard deviation used to generate data
θtrue = [μ,σ]

θ0 = [1,1] #initial guess
lb = [0.01,0.01] #lower bounds
ub = [2,2] #upper bounds

#Set random seed for reproducibility
Random.seed!(4107)

NormDist =  Normal(μ,σ) #normal distributuion with mean μ and standard deviation σ
Data = rand(NormDist,N) #generate N synthetic datapoints
llfun = θ -> loglikelihood(Normal(θ[1],θ[2]),Data) #loglikelihood function
MLE,MLL  = optimise(llfun,θ0,lb,ub) #compute maximum loglikelihood estimate (point-estimate of parameters), and corresponding loglikelihood value

#specify range for loglikelihood and scalar curvature surface plots
μrange = range(0.05,1.0,length=100)
σrange = range(0.2,1.0,length=100)

#Compute, normalise and plot loglikelihood surface
LogLikeSurface = [llfun([μval,σval]) for μval in μrange, σval in σrange]
NormalisedLogLikelihood = LogLikeSurface .- MLL
pltLogLikeSurface = heatmap(σrange,μrange,NormalisedLogLikelihood,xlabel = "σ", ylabel = "μ",right_margin = 20Plots.mm,bottom_margin = 5Plots.mm,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color=:oslo,clims=(-10,0))
pltLogLikeSurface = scatter!(pltLogLikeSurface,(MLE[2],MLE[1]))

CR = confidenceregion2D(MLE,MLL,llfun) #Compute confidence regions
CRplot = deepcopy(pltLogLikeSurface)
CRplot = plot!(CRplot,CR[:,2],CR[:,1],label="0.95 CR",linecolor=:magenta)

#Construct Fisher information matrix for the observation process
function Fisher(θ)
    FIM = N .* diagm([1/(θ[2]^2); 2/(θ[2]^2)])
    return FIM
end

#Compute and plot the scalar curvature surface (scalar curvature is third output of RiemannTensor function)
ScalarSurface = [RiemannTensor([μval,σval],5,Fisher)[3] for μval in μrange, σval in σrange]
ScalarSurfaceplot = heatmap(σrange,μrange,ScalarSurface,right_margin = 20Plots.mm,bottom_margin = 5Plots.mm,xlabel = "σ", ylabel = "μ",xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color=:davos,clims=(-0.11,0))

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
