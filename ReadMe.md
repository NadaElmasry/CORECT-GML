# Reimplementation of CORECT Paper (EMNLP 2023)

## To run preprocessing script

### on iemocap dataset
```
python preprocess.py --dataset="iemocap"
```
### on iemocap_4 dataset
```
python preprocess.py --dataset="iemocap_4"
```
## Training Example arguments:
``` 
python train.py --dataset="iemocap" --modalities="atv" --from_begin --epochs=50 --learning_rate=0.00025 --optimizer="adam" --drop_rate=0.5 --batch_size=10 --rnn="transformer" --use_speaker  --edge_type="temp_multi" --wp=11 --wf=5  --gcn_conv="rgcn" --use_graph_transformer --graph_transformer_nheads=7  --use_crossmodal --num_crossmodal=2 --num_self_att=3 --crossmodal_nheads=2 --self_att_nheads=2
```

### Training arguments explained:
For more detailed explaination and the available value for each argument please refer to `train.py`

- `--dataset`: Dataset name, with options ["iemocap", "iemocap_4"].
- `--data_dir_path`: Dataset directory path, default is "./data".
- `--from_begin`: Boolean flag indicating whether to start training from the beginning.
- `--device`: Computing device, default is "cuda".
- `--epochs`: Number of training epochs, default is 1.
- `--batch_size`: Batch size for training, default is 10.
- `--optimizer`: Name of optimizer, with options ["sgd", "rmsprop", "adam", "adamw"].
- `--scheduler`: Name of scheduler.
- `--learning_rate`: Learning rate for the optimizer, default is 0.00025.
- `--weight_decay`: Weight decay for the optimizer, default is 1e-8.
- `--max_grad_value`: Maximum gradient value, default is -1.
- `--drop_rate`: Dropout rate, default is 0.5.
- `--wp`: Past context window size, default is 11.
- `--wf`: Future context window size, default is 9.
- `--hidden_size`: Hidden size for the model, default is 100.
- `--rnn`: Type of RNN encoder cell, with options ["lstm", "transformer", "ffn"].
- `--class_weight`: Boolean flag indicating whether to use class weights in nll loss.
- `--modalities`: Modalities for the model, with options ["a", "t", "v", "at", "tv", "av", "atv"].
- `--gcn_conv`: Graph convolution layer, default is "rgcn".

- `--encoder_nlayers`: Number of encoder layers, default is 2.
- `--graph_transformer_nheads`: Number of attention heads in graph transformer, default is 7.
- `--use_highway`: Boolean flag indicating whether to use highway layers.
- `--seed`: Random seed, default is 24.
- `--data_root`: Data root folder, default is ".".
- `--edge_type`: Choices for edge construct type, with options ["temp_multi", "multi", "temp"].
- `--use_speaker`: Boolean flag indicating whether to use speakers attribute.
- `--no_gnn`: Boolean flag indicating whether to skip graph neural network.
- `--use_graph_transformer`: Boolean flag indicating whether to use graph transformer.
- `--use_crossmodal`: Boolean flag indicating whether to use crossmodal attention.
- `--crossmodal_nheads`: Number of attention heads in crossmodal attention block, default is 2.
- `--num_crossmodal`: Number of crossmodal blocks, default is 2.
- `--self_att_nheads`: Number of attention heads in self attention block, default is 2.
- `--num_self_att`: Number of self attention blocks, default is 3.
- `--tag`: Experiment tag, default is "normalexperiment".



## Testing Example arguments:
``` 
python eval.py --dataset="iemocap" --modalities="atv"
```

### Testing arguments explained:

For more detailed explaination and the available value for each argument please refer to `test.py`

- `--dataset`: Specifies the dataset name with choices ["iemocap", "iemocap_4"], default is "iemocap_4".

- `--data_dir_path`: Specifies the dataset directory path, default is "./data".

- `--device`: Specifies the computing device, default is "cpu".

- `--modalities`: Specifies the modalities to use with choices ["a", "at", "atv"], default is "atv".



### Important Notes:
- kept optimizer , eval , train, preprocess, and coach scripts largely similar to the original ones published by the author as they mainly contain command arguments and just pass them into the model.
- Kept all default model hyperparameters as defined by the author, also kept some hard-coded parameters that are not explained by the authors in the paper, for example the loss weights in `Classifier.py`.
- Multiheaded Attention code in the original paper was adapted from Fairseq code. In my implementation, I referenced the fairseq code and online resources. Same with the transformer code.