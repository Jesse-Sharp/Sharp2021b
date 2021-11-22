#=
    SIRODE.jl
    Here we form the right hand side of the SIR model system of differential equations, for use with the DifferentialEquations.jl ODE solvers.
=#

function SIRODE!(du,u,p,t)
    S,I,R = u
    (β,γ) = p
    du[1] = -β*S*I
    du[2] = β*S*I - γ*I
    du[3] = γ*I
end
