module DataModule

export generate_data, prepare_sequences

using Random

function generate_data(n_samples)
    # Create a more complex, periodic signal with multiple components
    x = range(0, 8pi, length=n_samples)  # Increased range for more periods
    
    # Combine multiple sine waves with different frequencies and amplitudes
    y = 0.8 .* sin.(x) .+                    # Main signal
        0.3 .* sin.(2.5 .* x) .+             # Higher frequency component
        0.2 .* sin.(0.5 .* x) .+             # Lower frequency component
        0.1 .* randn(length(x))              # Reduced noise for clearer pattern
    
    return Float32.(y)
end

function prepare_sequences(data, seq_length)
    x_data = []
    y_data = []
    # Use smaller stride for more training examples
    stride = 1  # You can adjust this value (1 to 5)
    
    for i in 1:stride:(length(data) - seq_length)
        sequence = reshape(data[i:i+seq_length-1], 1, :)
        push!(x_data, sequence)
        push!(y_data, data[i+seq_length])
    end
    
    X = cat(x_data..., dims=3)
    y = reshape(y_data, 1, :)
    return Float32.(X), Float32.(y)
end

end