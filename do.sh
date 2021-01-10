set -e
# python make_classcsv.py
# python make_source.py
python choice_traindata.py
python make_dataset.py
python textcnn_train.py
python textcnn_test.py