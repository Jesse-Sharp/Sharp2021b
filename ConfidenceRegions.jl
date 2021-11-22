#=
    ConfidenceRegions.jl
    Contains functions for producing confidence regions in two dimensional parameter space
=#

function confidenceregion2D(MLE,MLL,llfun;α=0.05,method=Heun(),dt=0.01,tN=1000,CritValue=quantile(Chisq(2),1.0-α)/2)
    llfun_grad = θ -> ForwardDiff.gradient(llfun,θ) #Function for gradient of loglikelihood
    #Fix one dimension to perform bisection in the other dimension to find a value on the desired confidence region
    Normllfun = θ -> llfun(θ).-MLL #Normalised loglikelihood function
    Normllfun1D = ϕ -> Normllfun([ϕ,MLE[2]])+CritValue
    BoundaryPt = [find_zero(Normllfun1D,[MLE[1],2*MLE[1]]), MLE[2]]
    # Move perpendicular to the gradient of the loglikelihood function
    function rhs(u,p,t)
            grd = llfun_grad(u)
            [-grd[2] ; grd[1]] / norm(grd)
    end
    # Stop once CR returns to BoundaryPt
    function callback(u,t,int)
        res = ((u[2] > BoundaryPt[2] > int.uprev[2]) || (u[2] < BoundaryPt[2] < int.uprev[2]))&&abs((u[1]-BoundaryPt[1])/u[1])<1e-2
        # determines that we have approximately returned to the original starting point, completing the confidence region
        return res
    end
    cb  = DiscreteCallback(callback,integrator -> terminate!(integrator))
    # Solve
    sol = solve(ODEProblem(rhs,BoundaryPt,(0.0,tN)),callback = cb,method,dt=dt,reltol=1e-5)
    CR   = hcat(sol.u...)'
    return CR
end
