# src/model.jl
module ModelModule

export create_rnn_model, train_model!

using Flux

function create_rnn_model(input_size, hidden_size, output_size)
    return Chain(
        Flux.RNN(input_size => hidden_size),
        Dense(hidden_size => output_size),
        x -> x[1, end, :]  # Take the last timestep output
    )
end

function train_model!(model, X, y; epochs=100, learning_rate=0.01)
    opt = Flux.setup(Adam(learning_rate), model)
    losses = Float64[]
    
    function compute_loss(m)
        ŷ = m(X)
        return Flux.mse(ŷ, vec(y))  # Flatten y to match the dimensions of ŷ
    end    
    
    for epoch in 1:epochs
        loss, grads = Flux.withgradient(compute_loss, model)
        Flux.update!(opt, model, grads[1])
        
        push!(losses, loss)
        
        if epoch % 10 == 0
            println("Epoch $epoch: Loss = $loss")
        end
    end
    
    return losses
end

end