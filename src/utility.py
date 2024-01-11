import json

import matplotlib.pyplot as plt
import pandas as pd
from haystack import Label, MultiLabel
from haystack.nodes.reader.base import BaseReader
from haystack.nodes.retriever.base import BaseRetriever
from haystack.pipelines import DocumentSearchPipeline, Pipeline


def evaluate_retriever(
    retriever: BaseRetriever, labels: MultiLabel | Label, topk_values=[1, 3, 5, 10, 20]
) -> pd.DataFrame:
    """Evaluate the retriever for different topk values
    The retriever can be `sparse` or `dense` retriever

    Args:
        retriever (BaseRetriever): the document retriever
        topk_values (list, optional): list of topk values. Defaults to [1,3,5,10,20].
    Returns:
        Dataframe contains the metrics calculated for each top_k value
    """
    topk_results = {}
    max_top_k = max(topk_values)

    # create the pipeline
    p = DocumentSearchPipeline(retriever=retriever)

    # run inference through all question-answer pair
    eval_result = p.eval(labels=labels, params={"Retriever": {"top_k": max_top_k}})

    # calculate metric for each top_k value
    for topk in topk_values:
        metrics = eval_result.calculate_metrics(simulated_top_k_retriever=topk)
        topk_results[topk] = {"recall": metrics["Retriever"]["recall_single_hit"]}

    return pd.DataFrame.from_dict(topk_results, orient="index")


def evaluate_reader(reader: BaseReader, labels: MultiLabel | Label) -> dict:
    """Calculate the ExactMatch and F1 score of the reader

    Args:
        reader (BaseReader): The reader
        labels (MultiLabel | Label): The labels

    Returns:
        dict: The score of each metric
    """
    score_keys = ["exact_match", "f1"]
    p = Pipeline()
    p.add_node(component=reader, name="Reader", inputs=["Query"])
    eval_result = p.eval(
        labels=labels,
        documents=[
            [label.document for label in multilabel.labels] for multilabel in labels
        ],
        params={},
    )
    metrics = eval_result.calculate_metrics(simulated_top_k_reader=1)

    return {k: v for k, v in metrics["Reader"].items() if k in score_keys}


def plot_retriever_eval(dfs: list[pd.DataFrame], retriever_names: list[str]):
    """Save the top-k recall values for each type of retriever

    Args:
        dfs (list[pd.DataFrame]): list of recall@k values for each type of retriever
        retriever_names (list[str]): list of type of retriever
    """
    fig, ax = plt.subplots()
    for df, retriever_name in zip(dfs, retriever_names):
        df.plot(y="recall", ax=ax, label=retriever_name)
    plt.xticks(df.index)
    plt.ylabel("Top-k Recall")
    plt.xlabel("k")
    plt.show()
    # plt.savefig("img/retriever_evaluation.png")


def plot_reader_eval(reader_eval):
    fig, ax = plt.subplots()
    df = pd.DataFrame.from_dict(reader_eval).reindex(["exact_match", "f1"])
    df.plot(kind="bar", ylabel="Score", rot=0, ax=ax)
    ax.set_xticklabels(["EM", "F1"])
    plt.legend(loc="upper left")
    plt.show()
    # plt.savefig("img/reader_evaluation.png")


def create_paragraphs(df: pd.DataFrame) -> list[dict]:
    """convert the dataframe to the SQuAD's paragraphs structure, this helps for fine-tuning the model for question-answering task

    Args:
        df (pd.DataFrame): the input dataframe to be converted

    Returns:
        list[dict]: list of SQuAD paragraphs
    """
    paragraphs = []
    id2context = dict(zip(df["review_id"], df["context"]))
    for review_id, review in id2context.items():
        qas = []
        # Filter for all question-answer pairs about a specific context
        review_df = df.query(f"review_id == '{review_id}'")
        id2question = dict(zip(review_df["id"], review_df["question"]))
        # Build up the qas array
        for qid, question in id2question.items():
            # Filter for a single question ID
            question_df = review_df.query(f"id == '{qid}'").to_dict(orient="list")
            ans_start_idxs = question_df["answers.answer_start"][0].tolist()
            ans_text = question_df["answers.text"][0].tolist()
            # Fill answerable questions
            if len(ans_start_idxs):
                answers = [
                    {"text": text, "answer_start": answer_start}
                    for text, answer_start in zip(ans_text, ans_start_idxs)
                ]
                is_impossible = False
            else:
                answers = []
                is_impossible = True
            # Add question-answer pairs to qas
            qas.append(
                {
                    "question": question,
                    "id": qid,
                    "is_impossible": is_impossible,
                    "answers": answers,
                }
            )
        # Add context and question-answer pairs to paragraphs
        paragraphs.append({"qas": qas, "context": review})

    return paragraphs


def convert_to_squad(dfs):
    for split, df in dfs.items():
        subjqa_data = {}
        # Create `paragraphs` for each product ID
        groups = (
            df.groupby("title")
            .apply(create_paragraphs)
            .to_frame(name="paragraphs")
            .reset_index()
        )
        subjqa_data["data"] = groups.to_dict(orient="records")
        # Save the result to disk
        with open(f"dataset/electronics-{split}.json", "w+", encoding="utf-8") as f:
            json.dump(subjqa_data, f)
