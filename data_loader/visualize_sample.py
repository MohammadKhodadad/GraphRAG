import os
import json
from utils.graph_generation.pipeline import graph_pipeline, sample_graph_pipeline


sampled_paths = sample_graph_pipeline('./data/chemrxiv_graph_v2.json',{1:10 })
print([sampled_paths[1][i][0][:3] for i in range(10)])
