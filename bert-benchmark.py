# import evaluate
# from tqdm import tqdm
# import timeit
# import pandas as pd
# from datasets import load_dataset
# from evaluate import evaluator
# from transformers import pipeline



# # Define the evaluation names you want to load
# evaluation_names = ["accuracy", "word_length"]  # Add more if needed

# # Define a function to load and evaluate an evaluation
# def load_and_evaluate(eval_name):
#     evaluation = evaluate.load(eval_name)
#     # You can perform additional processing or evaluation here if needed

# # Create a tqdm progress bar
# with tqdm(total=len(evaluation_names)) as pbar:
#     for eval_name in evaluation_names:
#         # Measure the execution time for loading and evaluating each evaluation
#         load_time = timeit.timeit(lambda: load_and_evaluate(eval_name), number=1)
#         pbar.update(1)
#         print(f"Load time for {eval_name}: {load_time:.2f} seconds")


# task_evaluator = evaluator("text-classification")

# # evaluate on different metrics
# eval_results = task_evaluator.compute(
#     model_or_pipeline="lvwerra/distilbert-imdb",
#     data=data,
#     metric=evaluate.combine(["accuracy", "recall", "precision", "f1"]),
#     label_mapping={"NEGATIVE": 0, "POSITIVE": 1}
# )
# print(eval_results)



# # load the models
# models = [
#     "xlm-roberta-large-finetuned-conll03-english",
#     "dbmdz/bert-large-cased-finetuned-conll03-english",
#     "elastic/distilbert-base-uncased-finetuned-conll03-english",
#     "dbmdz/electra-large-discriminator-finetuned-conll03-english",
#     "gunghio/distilbert-base-multilingual-cased-finetuned-conll2003-ner",
#     "philschmid/distilroberta-base-ner-conll2003",
#     "Jorgeutd/albert-base-v2-finetuned-ner",
# ]

# data = load_dataset("conll2003", split="validation").shuffle().select(1000)
# task_evaluator = evaluator("token-classification")

# results = []
# for model in models:
#     results.append(
#         task_evaluator.compute(
#             model_or_pipeline=model, data=data, metric="seqeval"
#             )
#         )

# df = pd.DataFrame(results, index=models)
# df[["overall_f1", "overall_accuracy", "total_time_in_seconds", "samples_per_second", "latency_in_seconds"]]