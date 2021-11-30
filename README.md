# Sigma

The code here is made publicly available for the research article "Self-Organizing to Learn Dynamical Hierarchies"
Refer to the article for details on the Sigma and the experiments performed.

## Usage example

To reproduce the experiment results, run:
```
$python main.py
$python TP_matrix_clustering.py
```

If the parameter `movie=True` and `map_dimensions=2` or `map_dimensions=3`, movies (MP4) of the dynamic of Sigma will be generated and saved in "movies_output" folder.

If the parameter `verbose=1` or `verbose=1`, experiment results will be saved in "result" folder in Excel file format (xlsx).

## Environments

All environments data are saved in the 'data' folder.

problem_index refers to the environments:

	IH: Imbalanced Hierarchy
	HB: Hierarchy with Branches
	IEH: Imbalanced with Extra Hierarchy
	DIH: Dynamic Imbalanced Hierarchy
	DCH: Dynamic Chunk Hierarchy
	EC2EH: Extra Chunk to Extra Hierarchy
	EH2EC: Extra Hierarchy to Extra Chunk
	DCS: Dynamic Chunk Swap

