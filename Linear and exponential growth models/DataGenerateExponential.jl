#=
    DataGenerateExponential.jl
    Produces synthetic data for the exponential model
=#

function DataGenerateExponential(θ,T,NperT,σ)
    Random.seed!(4107)
    if length(NperT) == 1  #equal number of observations per time-point
        Data = zeros(NperT*length(T))
        Time = zeros(NperT*length(T))
        for (i,t) in enumerate(T)
            NormDist =  Normal(ExponentialSolAllParams(θ,t),σ)
            Time[((i-1)*NperT+1):i*NperT] = t.*ones(NperT)
            #Data drawn from normal distribution with mean corresponding to the model behaviour
            Data[((i-1)*NperT+1):i*NperT] = rand(NormDist,NperT)
        end
    else
    #different number of observations per time-point
        Data = zeros(sum(NperT))
        Time = []
        for a in 1:length(NperT)
        append!(Time, T[a].*ones(NperT[a]))
        end
        for j in 1:sum(NperT)
            NormDist =  Normal(ExponentialSolAllParams(θ,Time[j]),σ)
            #Data drawn from normal distribution with mean corresponding to the model behaviour
            Data[j] = rand(NormDist)
        end
    end
return Time, Data
end
