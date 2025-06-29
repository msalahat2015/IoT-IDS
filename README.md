## Benchmarking AI models for IoT Network Intrusion Detection:An Analysis of Performance and Time Efficiency

COMP9340 – Computer Security Course - 2025

- Lecture: Prof. [Mohaiesn Daivad](https://www.cs.ucf.edu/~mohaisen/)
- Students: [Ruwa AbuHweidi](https://github.com/RuwaYafa) and [Monther Salahat](https://github.com/msalahat2015/IoT-IDS)

![Methodology Pipeline](IoT-Methodology.png)
---
* Our Project has two evaluation parts: Deep Learning and Machine Learning code are available on this repository [Monther Salahat](https://github.com/msalahat2015/IoT-IDS), and LLM code [Ruwa AbuHweidi](https://github.com/RuwaYafa/IoT-IDS-LLM)
* All Results are uploaded in repository [Dropbox](https://www.dropbox.com/scl/fo/7y8a8j7tko3da90sr6mco/AHpHYC95o7b65hER_MAKjKs?rlkey=2m6asv519j7w6li4d2c45ecbo&dl=0):
  * [Logs](https://www.dropbox.com/scl/fo/62zortvc8kuwud8ptn1n0/AEUulu3p5iGV0338tpijExU?rlkey=oaj70ynnmv2dtruwp5j4obs9v&dl=0).
  * [Saved models (Machine and Deep Learning)](https://drive.google.com/drive/folders/14io4lIMozrjQo1An5drDHKUwH6-ukY3X?usp=sharing).
  * [Checkpoints](https://www.dropbox.com/scl/fo/yg306y5df9y5eyjp9mz72/AKbDfF1MVBljfVSORMXKsC4?rlkey=wgv7mnhscnbvij32v12qd8lgo&dl=0) for trained LLM models.
---
## Project Structure 
<pre><code>
├── src/
│   ├── algorithms/
│   │   ├── _pycache_/
│   │   ├── base/
│   │   │   ├── _pycache_/
│   │   │   ├── _init_.py
│   │   │   └── base_algorithm.py
│   │   ├── custom_learning/
│   │   │   ├── _pycache_/
│   │   │   ├── stacking.py
│   │   │   └── voting.py
│   │   ├── deep_learning/
│   │   │   ├── _pycache_/
│   │   │   ├── DL-BiLstm.py
│   │   │   ├── _init_.py
│   │   │   ├── ann.py
│   │   │   ├── bilstm.py
│   │   │   ├── cnn.py
│   │   │   ├── dnn.py
│   │   │   ├── lstm.py
│   │   │   └── rnn.py
│   │   ├── machine_learning/
│   │   │   ├── _pycache_/
│   │   │   ├── _init_.py
│   │   │   ├── adaboost.py
│   │   │   ├── autoencoder.py
│   │   │   ├── balanced_svm.py
│   │   │   ├── decision_tree.py
│   │   │   ├── extra_trees_classifier.py
│   │   │   ├── gradient_boosting.py
│   │   │   ├── isolation_forest.py
│   │   │   ├── knn.py
│   │   │   ├── lightgbm.py
│   │   │   ├── localoutlierclass_lof.py
│   │   │   ├── logistic_regression.py
│   │   │   ├── mlp.py
│   │   │   ├── naive_bayes.py
│   │   │   ├── pca.py
│   │   │   ├── perceptron.py
│   │   │   ├── random_forest.py
│   │   │   ├── svm.py
│   │   │   ├── threshold.py
│   │   │   ├── weighted_svm.py
│   │   │   └── xgboost.py
│   │   └── reinforcement_learning/
│   │       ├── _pycache_/
│   │       ├── _init_.py
│   │       └── dqn.py
│   ├── utils/
│   │   └── _init_.py
│   ├── datasets/
│   │   ├── _pycache_/
│   │   ├── custom/
│   │   │   ├── _pycache_/
│   │   │   ├── cic_preprocessing.py
│   │   │   ├── _init_.py
│   │   │   ├── data_loader.py
│   │   │   └── preprocessing.py
│   │   └── evaluation/
│   │       ├── _pycache_/
│   │       ├── _init_.py
│   │       ├── metrics.py
│   │       └── visualization.py
│   ├── _init_.py
│   └── main.py
├── algorithms.json  # If this is a config file related to algorithms, it stays.
└── README.md
</code></pre>





