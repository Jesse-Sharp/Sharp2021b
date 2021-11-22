#=
    optimise.jl
    Optimisation routine for (e.g.) obtaining MLE
=#

function optimise(llfun,θ0,lb,ub;
    method = :LN_BOBYQA
    )
    #Convert Floats to one-element vectors
    if length(θ0) == 1
        θ0 = [θ0]
        lb = [lb]
        ub = [ub]
    end
    maxllfun = (θ,∂θ) -> llfun(θ) #Syntax requires derivative even with derivative free method
    #For NLopt syntax see https://github.com/JuliaOpt/NLopt.jl/blob/master/README.md
    opt = Opt(method,length(θ0))
    opt.max_objective = maxllfun #function we seek to maximise
    opt.lower_bounds = lb #lower bound
    opt.upper_bounds = ub #upper bound
    opt.local_optimizer = Opt(:LN_BOBYQA, length(θ0))
    sol = optimize(opt,θ0)
    MLE = sol[2] #maximum loglikelihood (point-estimate)
    MLL = sol[1] #maximum loglikelihood (value of the loglikleihood function at the MLE)
    return MLE, MLL
end
