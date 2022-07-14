task_name = "model"


config = {
    "dataset": "text",
    "model_name": task_name,
    "vocab_path": "./pretrained/glove/vocab.pkl",
    "gpu": "0",
    "dropout": 0.5,
    "patience": 6,
    "epochs": 15,
    "batch_size": 32,
    "pad_size": 32,
    "learning_rate": 5e-3,
    "save_result": True,
}