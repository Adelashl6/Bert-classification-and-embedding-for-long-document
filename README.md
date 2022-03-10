# BERT-Classification
Bert Classification for long documents

## Training
```python main.py --mode train --data_path PATH_TO_DATA --save_model --num_labels NUM_LABELS --save_result```

## Testing
```python main.py --mode test --data_path PATH_TO_DATA --num_labels NUM_LABELS --save_file PATH_TO_RESULTS --save_result```

## Inference
predict labels and extract bert embeddings
```python inference.py --data_path PATH_TO_DATA --num_labels NUM_LABELS --save_file_label PATH_TO_LABEL --save_file_embedding PATH_TO_EMBEDDING --save_result```

## Data Format

story_id  | raw_text | label
------------- | ------------- | -------------
integer  | string | integer
