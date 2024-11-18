## Data preparation

1. Min-cost flow instances:
    - Download the GOTO min-cost flow instances from [here](http://lime.cs.elte.hu/~kpeter/data/mcf/goto/) to folder `data/goto`. 
The instances are in the format of `*.min`. Transfer them to the format of `*.mcf` by running `scripts/min2mcf.py`.
    - Download the benchmark problems from [here](https://plato.asu.edu/ftp/lptestset/network/) to folder `data/mcf`.

2. Optimal transport instances:
   - Download the MNIST dataset from [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download-directory&select=train-images-idx3-ubyte) to folder `data/mnist`. 
   - Run `scripts/mnist2ot.py` to generate the optimal transport instances.

3. LP instances:
   - Most of the LP instances we used can be downloaded from [here](https://plato.asu.edu/ftp/lptestset/) to folder `data/lp`. 
   - There are also instances can be downloaded from the following links: 
[rail02](https://miplib2010.zib.de/miplib2010/rail02.php),
[shs1023](https://miplib2010.zib.de/miplib2010/shs1023.php),
[stp3d](https://miplib2010.zib.de/miplib2010/stp3d.php),
[karted, degme](http://old.sztaki.hu/~meszaros/public_ftp/lptestset/New/),
[graph20-20-1rand](https://miplib.zib.de/instance_details_graph20-20-1rand.html),
[graph20-80-1rand](https://miplib.zib.de/instance_details_graph20-80-1rand.html),
[graph40-20-1rand](https://miplib.zib.de/instance_details_graph40-20-1rand.html),
[graph40-40-1rand](https://miplib.zib.de/instance_details_graph40-40-1rand.html),
and [graph40-80-1rand](https://miplib.zib.de/instance_details_graph40-80-1rand.html). 
