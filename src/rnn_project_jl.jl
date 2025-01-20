module rnn_project_jl

include("data.jl")
include("model.jl")
include("visualization.jl")

using .DataModule
using .ModelModule
using .VisualizationModule

export generate_data, prepare_sequences
export create_rnn_model, train_model!
export plot_results

end
