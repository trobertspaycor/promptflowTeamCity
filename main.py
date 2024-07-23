from azure.identity import DefaultAzureCredential
import json
import os
import pandas as pd
from dotenv import load_dotenv

from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.evals.evaluators import (
    GroundednessEvaluator,
    HateUnfairnessEvaluator,
    RelevanceEvaluator,
    SelfHarmEvaluator,
    SimilarityEvaluator,
    ViolenceEvaluator,
)

from promptflow.client import PFClient
from promptflow.entities import AzureOpenAIConnection

# Get a pf client to manage connections
pf = PFClient()

load_dotenv()
api_key = os.getenv('AZURE_OPENAI_API_KEY')
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
model = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
version = os.getenv('AZURE_OPENAI_API_VERSION')

# Initialize an AzureOpenAIConnection object
connection = AzureOpenAIConnection(
    name="pf_test",
    api_key=api_key,
    api_base=endpoint,
    api_version=version
)

# Create the connection, note that api_key will be scrubbed in the returned result
result = pf.connections.create_or_update(connection)
print(result)

model_config = AzureOpenAIModelConfiguration(
    azure_endpoint=endpoint,
    api_key=api_key,
    azure_deployment=model,
)

project_scope = {
    "subscription_id": "77e69dbb-cf80-4ca0-a4f0-c9adf9550c37",
    "resource_group_name": "rg-contosoaihub",
    "project_name": "<<>>",
}

test_data_file = "test_questions.xlsx"


def run_quality_evaluators(data: pd.DataFrame):
    final_scores = []

    for index, row in data.iterrows():
        eval_dict = {}
        eval_dict['question'] = row['question']
        eval_dict['ground_truth'] = row['ground_truth']
        eval_dict['answer'] = row['answer']
        eval_dict['context'] = row['context']

        # Groundedness
        groundedness_eval = GroundednessEvaluator(model_config)
        groundedness_score = groundedness_eval(
            answer=row['answer'],
            context=row['context'],
        )
        eval_dict['groundedness'] = groundedness_score

        # Relevance
        relevance_eval = RelevanceEvaluator(model_config)
        relevance_score = relevance_eval(
            question=row['question'],
            answer=row['answer'],
            context=row['context'],
        )
        eval_dict['relevance'] = relevance_score

        # Similarity
        similarity_eval = SimilarityEvaluator(model_config)
        similarity_score = similarity_eval(
            question=row['question'],
            answer=row['answer'],
            ground_truth=row['ground_truth'],
        )
        eval_dict['similarity'] = similarity_score

        final_scores.append(eval_dict)

    return final_scores


def run_safety_evaluators(data: pd.DataFrame):
    safety_eval_scores = {}

    # Violence
    violence_eval = ViolenceEvaluator(project_scope)
    violence_score = violence_eval(question="What is the capital of France?", answer="Paris.")  # this does not work, auth issues
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
    data_df = pd.read_excel(test_data_file)

    quality_scores = run_quality_evaluators(data_df)

    with open("output.txt", "w") as file:
        for dictionary in quality_scores:
            quality_scores_json = json.dumps(dictionary)
            file.write(quality_scores_json + '\n')
