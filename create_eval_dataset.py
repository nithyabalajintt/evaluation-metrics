import json
from time import sleep

test_data = []


def create_eval_ds(agent, ground_truth_path):
    with open(ground_truth_path, 'r') as f:
        data = json.load(f)

        for obj in data:
            print(f'question:{obj["question"]}')
            resp = {'user_input': obj["question"], 'reference': obj["answer"]}
            # trigger the agent
            response = agent.run(obj["question"], markdown=True)

            relevant_docs = []
            for reference in response.extra_data.references:
                relevant_docs.extend([item['content'] for item in reference.references])

            resp['retrieved_contexts'] = relevant_docs
            resp['response'] = response.content

            test_data.append(resp)
            sleep(60)
    return test_data