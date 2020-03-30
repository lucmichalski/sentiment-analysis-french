from adaptnlp import LMFineTuner


def test_language_model_fine_tuner():
    train_data_file = "Path/to/train.csv"
    eval_data_file = "Path/to/test.csv"

    # Instantiate Finetuner with Desired Language Model
    finetuner = LMFineTuner(
        train_data_file=train_data_file,
        eval_data_file=eval_data_file,
        model_type="bert",
        model_name_or_path="bert-base-cased",
    )
    finetuner
