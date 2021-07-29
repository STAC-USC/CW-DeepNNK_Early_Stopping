# Use CW-DeepNNK early stopping in the training of a ConvNet
# add '--mode test' after training to evaluate the final model
python main.py --stopping cwdeepnnk --validation_percent 0 --criterion_freq 1 --patience 20 --knn_param 25 --interpol_queries 1.0
python main.py --stopping cwdeepnnk --validation_percent 0 --criterion_freq 10 --patience 2 --knn_param 25 --interpol_queries 1.0
python main.py --stopping cwdeepnnk --validation_percent 0 --criterion_freq 10 --patience 2 --knn_param 15 --interpol_queries 1.0
python main.py --stopping cwdeepnnk --validation_percent 0 --criterion_freq 10 --patience 2 --knn_param 15 --interpol_queries 0.5