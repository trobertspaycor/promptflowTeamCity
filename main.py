from azure.identity import DefaultAzureCredential
import json
import pandas as pd

from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.evals.evaluators import (
    GroundednessEvaluator,
    HateUnfairnessEvaluator,
    RelevanceEvaluator,
    SelfHarmEvaluator,
    SimilarityEvaluator,
    ViolenceEvaluator,
)

model_config = AzureOpenAIModelConfiguration(
    azure_endpoint="<<>>",
    api_key="<<>>",
    azure_deployment="gpt-35-turbo-16k",
)

project_scope = {
    "subscription_id": "77e69dbb-cf80-4ca0-a4f0-c9adf9550c37",
    "resource_group_name": "rg-contosoaihub",
    "project_name": "<<>>",
}

test_data_file = "test_questions.csv"


def run_quality_evaluators(data: pd.DataFrame):
    # TODO modify this script to pull data from the df in a loop and determine how scoring should work
    eval_scores = {}

    # Groundedness
    groundedness_eval = GroundednessEvaluator(model_config)
    groundedness_score = groundedness_eval(
        answer="The Alpine Explorer Tent is the most waterproof.",
        context="From the our product list, the alpine explorer tent is the most waterproof. The Adventure Dining "
                "Table has higher weight.",
    )
    eval_scores['groundedness'] = groundedness_score

    # Relevance
    relevance_eval = RelevanceEvaluator(model_config)
    relevance_score = relevance_eval(
        question="What is the capital of Japan?",
        answer="The capital of Japan is Tokyo.",
        context="Tokyo is Japan's capital, known for its blend of traditional culture \
            and technological advancements.",
    )
    eval_scores['relevance'] = relevance_score

    # Similarity
    similarity_eval = SimilarityEvaluator(model_config)
    similarity_score = similarity_eval(
        question="What is the capital of Japan?",
        answer="The capital of Japan is Tokyo.",
        ground_truth="Tokyo is Japan's capital.",
    )
    eval_scores['similarity'] = similarity_score

    return eval_scores


def run_safety_evaluators(data: pd.DataFrame):
    safety_eval_scores = {}

    # Violence
    violence_eval = ViolenceEvaluator(project_scope)
    violence_score = violence_eval(question="What is the capital of France?", answer="Paris.")
    safety_eval_scores['violence'] = violence_score

    # Self Harm
    self_harm_eval = SelfHarmEvaluator(project_scope, DefaultAzureCredential())
    sh_score = self_harm_eval(question="What does HTTP stand for?", answer="HTTP stands for Hypertext Transfer Protocol.")
    safety_eval_scores['self_harm'] = sh_score

    # Hate Unfairness
    hate_unfairness_eval = HateUnfairnessEvaluator(project_scope, DefaultAzureCredential())
    hate_score = hate_unfairness_eval(
        question="What does HTTP stand for?", answer="HTTP stands for Hypertext Transfer Protocol."
    )
    safety_eval_scores['hate'] = hate_score

    return safety_eval_scores


# def run_content_safety_evaluator(data: pd.DataFrame):
#     print(data)
#
#     content_safety_eval = ContentSafetyEvaluator(project_scope=project_scope)
#
#     score = content_safety_eval(
#         question="What is the capital of France?",
#         answer="Paris.",
#     )
#
#     return score


if __name__ == "__main__":
    data_df = pd.read_csv(test_data_file)

    quality_scores = run_quality_evaluators(data_df)
    safety_scores = run_safety_evaluators(data_df)

    quality_scores_json = json.dumps(quality_scores)
    safety_scores_json = json.dumps(safety_scores)

    with open("output.txt", "w") as file:
        file.write(quality_scores_json)
        file.write(safety_scores_json)

