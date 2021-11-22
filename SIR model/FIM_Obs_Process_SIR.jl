#=
    FIM_Obs_Process_SIR.jl
    Fisher information for the observation process of the SIR model where S, I and R are observed
=#

function FIM_Obs_Process_SIR(σ,T,N,m)
if length(N)==1
    N = N*ones(length(T))
end
M = zeros(2*length(T))
    for i in 1:length(T)
        M[2*i-1] = N[i]/σ^2
        M[2*i] = 2*N[i]/σ^2
    end
mM = Vector{Float64}(undef, 0)
for i in 1:m
    append!(mM,M)
end
    diagm(mM)
end
