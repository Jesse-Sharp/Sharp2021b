#=
    SIRdataplot.jl
    Produces plot of Murray influenza data with calibrated SIR model.
=#

#Initialisation (packages)
using CSV
using DataFrames
using Plots
using DifferentialEquations
#Initialisation (local functions and plot settings)
include("SIRSolAllParams.jl")
include("SIRODE.jl")
gr()
default(fontfamily="Arial")
SIRdata = CSV.read("MurrayData.csv",DataFrame) #load data from CSV
pop = 763 #total population is known (Murray, Mathematical Biology: https://link.springer.com/book/10.1007/b98868)
pltdata = scatter(SIRdata.time,SIRdata.Infected/pop,legend=:none,label="Data", markersize = 8,markercolor = :red,markerstrokecolor = :red,grid=:off)

#True values used to generate data (scaled based on total population to give proportions)
β = 2.18e-3*pop #rate of infection
γ = 0.4404  #rate of recovery
#Initial proportions
S₀ = 762/pop #Susceptible
I₀ = 1/pop #Infected
R₀ = 0/pop #Recovered
IC = [S₀;I₀;R₀]

#Compute SIR model solution
function SIRSol(θ,t,IC)
    (β,γ) = θ[1:2]
    sol=SIRSolAllParams([β;γ;IC],t)
    return sol
end

#Add SIR solution to data plot
SIR_sol= SIRSol([β;γ],SIRdata.time[end],IC)
plot!(pltdata,SIR_sol,label = ["S" "I" "R"],color = [:blue :red :green],legend=:right,xlabel = "Time (days)", ylabel = "Population proportion",box = :on,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,linewidth =2)
plot!(pltdata,framestyle=:box)
plot!(pltdata,xticks= [0,2,4,6,8,10,12,14] ,xlims = (0,14.2))
