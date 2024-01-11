import os

from haystack.nodes.reader import FARMReader

from src.data import load_dataset
from src.utility import convert_to_squad

MODEL_CKPT = "deepset/minilm-uncased-squad2"


def main():
    print("Loading the subjqa dataset")
    subjqa = load_dataset("subjqa", name="electronics")
    dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}

    print("Initializing the reader")
    max_seq_length, doc_stride = 384, 128
    reader = FARMReader(
        model_name_or_path=MODEL_CKPT,
        progress_bar=False,
        max_seq_len=max_seq_length,
        doc_stride=doc_stride,
        return_no_answer=True,
    )

    if not os.path.exists("dataset/electronics-train.json"):
        print("Convert subjqa dataset to SQuAD format")
        convert_to_squad(dfs)

    print("Training..")
    train_filename = "electronics-train.json"
    dev_filename = "electronics-validation.json"

    reader.train(
        data_dir="dataset",
        use_gpu=False,
        n_epochs=1,
        batch_size=16,
        train_filename=train_filename,
        dev_filename=dev_filename,
    )
    print("Finished")


if __name__ == "__main__":
    main()
