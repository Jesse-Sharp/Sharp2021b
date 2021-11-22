#=
    Logisticdataplot.jl
    Produces plot of coral coverage data with calibrated logistic model.
=#

#Initialisation (packages)
using CSV
using DataFrames
using Plots
#Initialisation (local functions and plot settings)
logisticdata = CSV.read("Logistic_data.csv",DataFrame)
gr()
default(fontfamily="Arial")
pltdata = scatter(logisticdata.time/365.25,logisticdata.data,legend=:none,label="Data", markersize = 8,markercolor = :olive,markerstrokecolor=:olive)

#True values used to generate data
r = 0.0025*365.25 #rate of growth
C = 0.7237 #initial coverage (C(0))
K = 79.74 #carrying capacity

#Compute logistic model solution
function LogisticSol(θ,t,K)
    (r,C0) = θ
    Logistic = K*C0/(C0+(K-C0)*exp(-r*t))
    return Logistic
end

#Add logistic solution to data plot
Tspan = range(0,4050/365.25,length = 100)
Logistic_sol=zeros(length(Tspan))
for (i,t) in enumerate(Tspan)
Logistic_sol[i] = LogisticSol([r,C],t,K)
end
plot!(pltdata,(Tspan,Logistic_sol),label = "Logistic model fit",color = :green,legend=:bottomright,xlabel = "Time (years)", ylabel = "Population coverage (%)",box = :on,grid=:off,xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,legendfontsize=16,linewidth =2)
plot!(pltdata,framestyle=:box)
