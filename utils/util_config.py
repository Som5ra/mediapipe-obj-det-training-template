import importlib.util 


def load_config(config_path):

    spec = importlib.util.spec_from_file_location(
        name="my_module",  # note that ".test" is not a valid module name
        location = config_path,
    )
    my_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_module)
    return my_module

# if __name__ == "__main__":
#     config = load_config("configs/test.py")
#     print(config.epochs)
#     print(config.train_dataset_path)
#     print(config.validation_dataset_path)
#     print(config.weights_save_dir)
#     print(config.export_fp16)