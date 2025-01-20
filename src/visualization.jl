module VisualizationModule

export plot_results

using Plots

function plot_results(actual_data, predictions, losses)
    # Flatten predictions to match the shape of actual_data
    predictions = vec(predictions)  # Convert to a 1D vector if necessary

    # Training Loss Plot
    p1 = plot(losses, 
        title="Training Loss Over Time", 
        xlabel="Epoch", 
        ylabel="MSE Loss",
        label="Training Loss",
        linewidth=2,
        color=:blue,
        background_color=:white,
        grid=true,
        gridstyle=:dash,
        gridalpha=0.2)
    
    # Prediction vs Actual Plot
    p2 = plot(1:length(actual_data),
        [actual_data predictions],
        label=["Actual" "Predicted"],
        linewidth=[2 2],
        color=[:blue :red],
        alpha=[1.0 0.7],
        title="Time Series Prediction",
        xlabel="Time Step",
        ylabel="Value",
        background_color=:white,
        grid=true,
        gridstyle=:dash,
        gridalpha=0.2)
    
    # Add a zoomed-in section
    zoom_start = 200
    zoom_length = 100
    p3 = plot(zoom_start:zoom_start+zoom_length,
        [actual_data[zoom_start:zoom_start+zoom_length] predictions[zoom_start:zoom_start+zoom_length]],
        label=["Actual" "Predicted"],
        linewidth=[2 2],
        color=[:blue :red],
        alpha=[1.0 0.7],
        title="Zoomed Prediction (100 steps)",
        xlabel="Time Step",
        ylabel="Value",
        background_color=:white,
        grid=true,
        gridstyle=:dash,
        gridalpha=0.2)
    
    # Combine plots
    final_plot = plot(p1, p2, p3,
        layout=(3,1),
        size=(800,1000),
        margin=5Plots.mm)
    
    return final_plot
end

end