# WNFG-SSGCNet-ADMM
This repository holds the source code of SSGCNet model. The code consists of three files (data, result, save) and five .py files(utils.py, optimizer.py, model.py, method.py and main.py).

data file
Due to the size limitation of upload file, we provide website of the Bonn dataset：http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3&changelang=3. We upload AET.mat and train_label_bonn.mat as a example in data file.

result file
The results of the model will be written in the result file as a .txt file. It include the best accuracy of different pruning rate.

save file
The save file include all the records of training model.

utils.py
The admm pruning functions were written in this file.

optimizer.py
To prevent prunned weights from updated by optimizer, I modified Adam (named PruneAdam).

The utils.py and optimizer.py, I refer to the pytorch implementation of DNN weight prunning with ADMM described in [**A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers**](https://arxiv.org/abs/1804.03294).

model.py
The SSGCNet framework in this file.

method.py
All the graph representation method are in this file(include: VG, LVG, HVG, LHVG, OG, WOG, WNFG).

main.py
The data load, running function are in this file. 

To use this classifier, please make sure that you have pytorch, Python 3.8 and all the packages we have used installed.

Please take the following two steps.

Step 1. Change the path in line 125 of main.py to the path of input data.

Step 2. Run the command in your command council. 
$ python main.py
