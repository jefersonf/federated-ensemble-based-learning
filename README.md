
---

This code aims to simulate our experiments with ensemble-based learning approach that was proposed in the "Ensemble-based Federated Learning" paper submitted to HPE TechCon 2020.

---

## Ensemble-based Federated Learning

Basically, our approach is based on model training done in a federated way by using model agnostic technique. In the article, we tried to address an open problem which is the possibility of having FL clients that train heterogeneous models, especially focusing on non-neural models (i.e. classical ML algorithms)

We restricted our experiments to cross-silo federated settings but we also not limited to investigate cross-device federation settings by trying to scaling up the number of nodes (Which can be extremely high!).

In our collaborative training routine there are two main events:
1. Clients send their models to a server responsible for maintaining and managing an ensemble model.
2. Clients receive an ensemble model and judge whether it is better to continue using their own model or use the shared ensemble model.

There are then two levels of model updates, local and global. The global model refers to the ensemble model and the local to client/node models. We always updated local and global models in our simulations when new data arrives at the local nodes. We could thus have different updating policies. For example, the frequency of new data arriving at different clients is different, so might be the frequency of local model updates. The global model could be updated, in turn, always that there is a change in performance in any local model. So, the point is that this can be easily customizable, according to the desired strategy. Because of the simpler/lightweight models the frequency of updates can be higher. Arrival of data can result in continuous updates and tweaking, requiring atomic access to enable consistency of updates.


<!--

Dejan: 
How frequently do we update models and is there any concurrency in updates that requires atomic access(e.g. multiple updates coming to the same node at the same time). If not, coud you envision such use case?

Leandro:
There are two levels of model updates, local and global. We could thus have different updating policies. For example, the frequency of new data arriving at different hospitals is different, so might be the frequency of local model updates. The global model could be updated, in turn, always that there is a change in performance in any local model. We always updated local and global models in our simulations when new data arrives at the local nodes. My point is that this can be easily customizable, according to the desired strategy. Does that answer your question?

Dejan:
This is exactly what I was interested in and you answered my question. This is the reason why I like your work as a use case. Because of the simpler models the frequency of updates can be higher. Arrival of data can result in continuous updates and tweaking, requiring atomic access to enable consistency of model updates.

-->

## Setting Up

We recommend using a test environment such as `venv`, `virtualenv` or any other to install dependencies.


`pip3 install -r requirements.txt`

## Usage 

To execute the whole experiment, some options need to be set.

Option | Description | Default setting
- | - | - |
`--rounds` | Communication rounds | 10
`--epochs` | Client training epochs | 5
`--clients` | Number of participating clients/nodes. For configurations with more than seven clients (ie `--clients > 7`) model types will repeat.  | 3
`--data-path` | Path to training data |
`--data-split` | Train/Validation/Test data split | 0.8, 0.1, and 0.1, respectively
`--dirichlet-alpha` | Alpha value of Dirichlet distribution of training data | 100
`--target-feature` | Target feature name to predict | 
`--data_distrib_mode` | Data distribution mode over rounds | uniform
`--fedavg` | Sets Federated Averaging Algorithm (FedAvg) | Ensemble-based Learning mode
`--model_type` | Define which NN model will be used for federated task | A
`--lr` | lr for optimizer function | 0.01
`--train_batch_size` | batch size for train NN model | 32
`--evaluate_batch_size` | batch size for test/val NN model | 64
`--enable_grouping` | Enables the grouping of models of the same arch | False
`--model_alocation` | Set to modify the beginning of the circular list | 0
`--seed` | Random seed value | 1
`-v, --verbose` | Sets verbose mode | False
`--log-dir` | Directory path to save experiment reports | 


## Demo

### Using Shelter Animal Outcomes Dataset

The setting is 3 clients with normally distributed training data over 10 rounds. 

FedEL mode.
```
python3 main.py --data-path datasets/shelter.csv --log-dir ./reports/shelter --verbose

```

FedAvg mode.
```
python3 main.py --fedavg --data-path datasets/shelter.csv --log-dir ./reports/shelter --verbose

```
