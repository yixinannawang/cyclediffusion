from datasets import load_dataset


ds = load_dataset("lambdalabs/pokemon-blip-captions")
ds = ds["train"].train_test_split(test_size=0.1)
ds_train = ds["train"]
ds_test = ds["test"]