#=
    FIM_Obs_Process_SIR_obsIonly.jl
    Fisher information for the observation process of the SIR model where only I is observed
=#

function FIM_Obs_Process_SIR_obsIonly(σ,T,N)
if length(N)==1
    N = N*ones(length(T))
end
M = zeros(2*length(T))
    for i in 1:length(T)
        M[2*i-1] = N[i]/σ^2
        M[2*i] = 2*N[i]/σ^2
    end
    return diagm(M)
end
