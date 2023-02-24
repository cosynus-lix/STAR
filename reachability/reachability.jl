using NeuralVerification
using LazySets

function reachability(alg::String, input_low::Array, input_high::Array, 
    output_low::Array, output_high::Array, network::String)
    
    if (cmp(alg, "Ai2") == 0)
        solver = Ai2()
    elseif (cmp(alg, "ConvDual") == 0)
        solver = ConvDual()
    elseif (cmp(alg, "ExactReach") == 0)
        solver = ExactReach()
    elseif (cmp(alg, "MaxSens") == 0)
        solver = MaxSens()
    end
    
    nnet = read_nnet(network)
    input_set  = LazySets.Hyperrectangle(low = input_low, high = input_high)
    output_set = LazySets.Hyperrectangle(low = output_low, high = output_high)
    problem = Problem(nnet, input_set, output_set)

    result = solve(solver, problem)
    R = result.reachable[1]
    
    if (cmp(alg, "Ai2") == 0)
        R = LazySets.Approximations.box_approximation(R)
    end

    low = vertices_list(R)[end]
    high = vertices_list(R)[1]

    return low, high
end

function compute_grad(input_low::Array, input_high::Array, network::String)

    nnet = read_nnet(network)
    input  = LazySets.Hyperrectangle(low = input_low, high = input_high)
    LG, UG = NeuralVerification.get_gradient_bounds(nnet, input)
    feature, monotone = NeuralVerification.get_max_smear_index(nnet, input, LG, UG)

    return feature-1
end
