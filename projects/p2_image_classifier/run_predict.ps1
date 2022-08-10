python predict.py "./test_images/cautleya_spicata.jpg" keras_model_1.h5
python predict.py "./test_images/hard-leaved_pocket_orchid.jpg" keras_model_1.h5
python predict.py "./test_images/orange_dahlia.jpg" keras_model_1.h5 --top_k 10
python predict.py "./test_images/wild_pansy.jpg" keras_model_1.h5 --category_names "./label_map.json"