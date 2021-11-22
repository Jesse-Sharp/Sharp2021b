#=
    SIR_Infer_Params_beta_gamma.jl
    Produces inference and information geometry results for the SIR model where S, I and R are observed, estimating θ = (β,γ)
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
include("../InfoGeo2D.jl")
include("../ConfidenceRegions.jl")
include("DataGenerateSIR.jl")
include("SIRSolAllParams.jl")
include("SIRODE.jl")
include("FisherSIR.jl")
include("SIR_Jacobian.jl")
include("SIR_partials.jl")
include("FIM_Obs_Process_SIR.jl")
include("../EquidistantCircumferencePoints.jl")
gr()
default(fontfamily="Arial")

TimePoints = [4,7,10] #timepoints at which to to generate synthetic data
NperT = [3,3,3] #number of observations at each time in TimePoints
FinalTime = 14 #final time for the simulation

pop = 763 #total population is known (Murray, Mathematical Biology: https://link.springer.com/book/10.1007/b98868)
#True values used to generate data (scaled based on total population to give proportions)
β = 2.18e-3*pop #rate of infection
γ = 0.44036 #rate of recovery
#Initial proportions
S₀ = 762/pop #Susceptible
I₀ = 1/pop #Infected
R₀ = 0/pop #Recovered
σ = 0.03 #standard deviation of synthetic data
θtrue = [β,γ]
Ntime,Ndata = DataGenerateSIR([β,γ,S₀,I₀,R₀],TimePoints,NperT,σ) #generate synthetic data
#Initial guesses
β0 = 1.0 #rate of infection
γ0 = 0.5 #rate of recovery
θ0 = [β0,γ0]
# Bounds for optimisation to determine MLE
β_min = 0.0 #lower bound on β
β_max = 2.0 #upper bound on β
γ_min = 0.0 #lower bound on γ
γ_max = 1.0 #upper bound on γ
lb = [β_min,γ_min] #lower bounds
ub = [β_max,γ_max] #upper bounds

#Solution to the SIR model as a function of the parameters, time and initial condition
function SIRSol(θ,t,IC)
    (β,γ) = θ
    sol=SIRSolAllParams([β;γ;IC],t)
    return sol
end

#Loglikleihood function
llfun = θ -> sum([ loglikelihood(MvNormal(SIRSol(θ[1:2],t,[S₀;I₀;R₀])(t),σ^2*I),[Ndata[i,:]]) for (i,t) in enumerate(Ntime)])
#Compute MLE and MLL
MLE,MLL  = optimise(llfun,θ0,lb,ub)

#PLot solution to model where parameters are given by the MLE
SIRSolMLE = SIRSol(MLE,FinalTime,[S₀;I₀;R₀])
pltsol=plot(SIRSolMLE,label=["S fit" "I fit" "R fit"],linecolor=[:blue :red :green])
scatter!(pltsol,Ntime, Ndata,label="Data",xlabel = "Time (days)",ylabel = "Population proportion",title = "SIR infer beta gamma",markercolor=[:blue :red :green],markersize = 5) #add synthetic data to plot
#Add solution based on true parameters to the plot
SIR_sol= SIRSol([β,γ],FinalTime,[S₀;I₀;R₀])
plot!(pltsol,SIR_sol,label = ["S true" "I true" "R true"],xlabel = "Time (years)",linecolor=[:blue :red :green],linestyle = :dash, markerstrokewidth = 1.5,legend=:left,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=14,box = :on, grid = :false,linewidth =2)

#Specify range for loglikelihood and scalar curvature surface plots
βrange = range(1.55,1.75,length=100)
γrange = range(0.4,0.55,length=100)

#Compute, normalise and plot loglikelihood surface
LogLikeSurface = [llfun([βval,γval]) for βval in βrange, γval in γrange]
NormalisedLogLikelihood = LogLikeSurface .- MLL
pltLogLikeSurface = heatmap(γrange,βrange,NormalisedLogLikelihood,xlabel = "γ", ylabel = "β",right_margin = 20Plots.mm,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color = :oslo,clims=(-10,0))
pltLogLikeSurface = scatter!(pltLogLikeSurface,(MLE[2],MLE[1]))

#Compute and plot confidence regions
CR = confidenceregion2D(MLE,MLL,llfun)
CRplot = deepcopy(pltLogLikeSurface)
CRplot = plot!(CRplot,CR[:,2],CR[:,1],label="0.95 CR",linecolor = :magenta, linewidth = 2)

#Construct Fisher information matrix for the SIR model where 3 popultion proportions (S, I and R) are observed, estimating θ = (β,γ)
function Fisher(θ)
    FIM = Fisher_βγ([θ;S₀;I₀;R₀],σ,TimePoints,NperT,3)
    return FIM
end

#Compute and plot the scalar curvature surface (scalar curvature is third output of RiemannTensor function)
ScalarSurface = [RiemannTensor([βval,γval],5,Fisher)[3] for βval in βrange, γval in γrange]
ScalarSurfaceplot = heatmap(γrange,βrange,ScalarSurface,xlabel = "γ", ylabel = "β",right_margin = 20Plots.mm,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,color=:lajolla,clims = (0,0.0015))

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
