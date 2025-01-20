using Pkg; Pkg.activate(".")
using rnn_project_jl
using Random
using Plots  # Import Plots to enable `savefig`

# Set random seed
Random.seed!(123)

# Set parameters
n_samples = 2000          # More samples for better training
seq_length = 50           # Longer sequence length to capture patterns
hidden_size = 64          # Larger hidden size for more capacity
epochs = 200              # More epochs for better convergence
learning_rate = 0.005     # Slightly lower learning rate for stability

# Generate and prepare data
data = generate_data(n_samples)
X, y = prepare_sequences(data, seq_length)

# Create and train model
model = create_rnn_model(1, hidden_size, 1)
losses = train_model!(model, X, y, epochs=epochs, learning_rate=learning_rate)

# Make predictions
predictions = model(X)

# Plot results
p = plot_results(data[seq_length+1:end], predictions, losses)
savefig(p, "results.png")  # Save the plot to a file
