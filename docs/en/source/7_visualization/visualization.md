## Visualization (New Feature!)

Galvatron Memory Visualizer is an interactive tool for analyzing and visualizing memory usage in large language models. Based on the Galvatron memory cost model, this tool provides users with intuitive visual representations of memory allocation for different model configurations and distributed training strategies.


<div align=center> <img src="../_static/visualizer-demo.gif" width="800" /> </div>

### Key Features

- **Interactive Memory Visualization**: View memory allocation with interactive treemap visualization
- **Memory Distribution Analysis**: Analyze memory usage by category with bar charts and proportion views
- **Distributed Training Strategies**: Configure tensor parallelism, pipeline parallelism, and other distribution strategies
- **Real-time Memory Estimation**: Get instant memory usage feedback when changing parameters
- **Bilingual Support**: Full Chinese and English interface support
- **Configuration Upload**: Import Galvatron configuration files for precise memory analysis

### Memory Categories

The visualizer analyzes and displays memory usage across several categories:

- **Activation Memory**: Memory used for storing activations during the forward pass
- **Model States**: Combined memory for parameters, gradients, and optimizer states
  - **Parameter Memory**: Memory used to store model parameters
  - **Gradient Memory**: Memory used for gradients during backpropagation
  - **Optimizer Memory**: Memory used by optimizer states
  - **Gradient Accumulation**: Memory used for gradient accumulation in multi-step updates

### Installation

#### Online Usage

Visit [Galvatron-Visualizer](http://galvatron-visualizer.pkudair.site/) to use the online version.

#### Run Locally

1. Clone the repository

	```bash
	git clone https://github.com/PKU-DAIR/Hetu-Galvatron.git
	cd Hetu-Galvatron
	git checkout galvatron-visualizer
	cd galvatron-visualizer
	```

2. Install dependencies

	```bash
	npm install
	```

3. Start the development server

	```bash
	npm start
	```

4. Open [http://localhost:3000](http://localhost:3000) to view the application

### Usage

1. **Select a Configuration**: Choose a predefined model or upload a configuration file
2. **Adjust Parameters**: Modify model parameters in the config panel
3. **View Memory Analysis**: Observe memory allocation in the treemap visualization
4. **Analyze Distributions**: Use the bar chart and proportion views to understand memory usage patterns