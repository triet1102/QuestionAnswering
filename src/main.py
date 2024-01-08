from datasets import get_dataset_config_names, load_dataset


def main():
    # https://huggingface.co/datasets/subjqa
    domains = get_dataset_config_names("subjqa")
    print(f"List of all domains in `Subjqa` dataset: {domains}\n")

    # load the electronics domain of the dataset
    subjqa = load_dataset(path="subjqa", name="electronics")
    print(f"Example of an answer:\n{subjqa['train']['answers'][1]}\n")

    # get the train/validation/test dataframes
    dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}
    for split, df in dfs.items():
        print(f"Number of questions in {split}: {df['id'].nunique()}")

    # title:                  The Amazon Standard Identification Number (ASIN) associated with each product
    # question:               The question
    # answers.text:    The span of text in the review labeled by the annotator
    # answers.answer_start:   The start character index of the answer span
    # context:                The customer review
    qa_cols = ["title", "question", "answers.text", "answers.answer_start", "context"]
    sample_df = dfs["train"][qa_cols].sample(2, random_state=7)
    print(f"2 random samples from training set:\n{sample_df}\n")

    # get a sample answer
    start_idx = sample_df["answers.answer_start"].iloc[0][0]
    end_idx = start_idx + len(sample_df["answers.text"].iloc[0][0])
    print(f"Question: {sample_df['question'].iloc[0]}")
    print(f"Answer: {sample_df['context'].iloc[0][start_idx:end_idx]}")

    # get types of frequent asked questions
    counts = {}
    question_types = ["What", "How", "Is", "Does", "Do", "Was", "Where", "Why"]
    for q in question_types:
        counts[q] = dfs["train"]["question"].str.start_with(q).value_counts()[True]


if __name__ == "__main__":
    main()
