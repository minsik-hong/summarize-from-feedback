import json
import os
from dataclasses import dataclass, field

import blobfile as bf
import numpy as np
import torch

from summarize_from_feedback.datasets import jsonl_encoding
from summarize_from_feedback.query_response_model import ModelSpec
from summarize_from_feedback.reward_model import RewardModel
from summarize_from_feedback.task_data import make_jsonl_samples_iter
from summarize_from_feedback.tasks import TaskHParams
from summarize_from_feedback.utils import Timer, hyperparams
from summarize_from_feedback.utils.assertions import assert_shape_eq, assert_eq
from summarize_from_feedback.utils.logging_utils import setup_logging_with_pacific_tz
from summarize_from_feedback.utils.torch_utils import to_numpy

"""
Evaluates a reward model on a set of query-responses examples. The output will contain the same
json data as the input along with an extra key containing the predicted reward.
"""


@dataclass
class HParams(hyperparams.HParams):
    reward_model_spec: ModelSpec = field(default_factory=ModelSpec) # 사용할 reward model의 사양
    task: TaskHParams = field(default_factory=TaskHParams) # 태스크 정의 (쿼리 길이 등 포함)
    input_path: str = None  # Should contain files samples.0.jsonl, samples.1.jsonl, ... 평가 대상 샘플 jsonl 파일 위치
    fp16_activations: bool = True # float16 연산 활성화 여부
    output_key: str = "predicted_reward" # 출력 JSONL에 쓸 reward key 이름


def main(H: HParams):
    layout = H.reward_model_spec.run_params.all_gpu_layout() # GPU 레이아웃 설정

    reward_model = RewardModel(task_hparams=H.task, spec=H.reward_model_spec, layout=layout) # RM 초기화

    # 로깅 설정 시작
    setup_logging_with_pacific_tz()

    act_dtype = torch.float16 if H.fp16_activations else torch.float32

    results_dir = bf.join(
        os.environ.get("OUTPUT_DIR", os.path.join("/tmp/jobs", os.getenv("JOB_NAME"))), "results"
    )
    bf.makedirs(results_dir) 

    if layout.is_logging_rank:
        with open(bf.join(results_dir, "task_hparams.json"), "w") as f:
            json.dump(H.task.to_json(), f)
        with open(bf.join(results_dir, "hparams.json"), "w") as f:
            json.dump(H.to_json(), f)
    # 로깅 설정 끝 
    
    # Creates files for printing. Only the replica root prints the files
    output_file_name = os.devnull
    if layout.is_replica_root:
        fname = f"samples.{layout.replica_idx}.jsonl"
        output_file_name = bf.join(results_dir, fname)
        print(f"Outputs will be written to {output_file_name}")
        
    # 출력 파일 설정 

    # H.input_path 위치에서 samples.0.jsonl, samples.1.jsonl, ... 등을 읽어드림 
    input_iter = make_jsonl_samples_iter(H.input_path, layout=layout)
    
    replica_rewards = []

    with open(output_file_name, "a") as out_f:
        input_idx = 0
        for input in input_iter:
            with Timer() as timer:
                query_tokens = torch.tensor(input["context_tokens"])
                assert_shape_eq(
                    query_tokens, (H.task.query.length,), "Context tokens shape mismatch"
                )
                response_tokens = torch.tensor(input["sample_tokens"])
                assert_eq(response_tokens.dim(), 2)

                n_responses = response_tokens.size(0)

                # context와 response를 모델에 넣어 reward를 계산
                results = reward_model.reward(
                    query_tokens=query_tokens.unsqueeze(0),
                    response_tokens=response_tokens.unsqueeze(0),
                    act_dtype=act_dtype,
                )

                # results["reward"]는 shape (1, N) → reshape해서 (N,)로 만듦
                rewards = to_numpy(results["reward"].reshape((n_responses,)))

                if layout.is_replica_root:
                    # 나중에 평균, 표준편차 등을 계산하기 위해 리스트에 저장
                    replica_rewards.append(rewards)
                    # reward는 predicted_reward라는 key로 붙어서 저장 
                    output = {**input, H.output_key: rewards}
                    out_f.write((json.dumps(jsonl_encoding.encode_example(output)) + "\n"))
            input_idx += 1
            if layout.is_replica_root:
                print(f"Batch {input_idx}.  Took {timer.interval} seconds")

        if layout.is_replica_root:
            print(f"Wrote {input_idx} batches to {output_file_name}")

            replica_rewards = np.stack(replica_rewards, axis=0)
            all_rewards = reward_model.dp_comm.mpi_all_gather(replica_rewards, "rewards")
            if layout.replica_idx == 0:
                all_rewards = np.concatenate(all_rewards, axis=0)
                print(f"Mean reward: {all_rewards.mean():.3f}")
                if all_rewards.shape[1] > 1:
                    print(f"Stddev within a query: {all_rewards.std(axis=1, ddof=1).mean():.3}")
                print(f"Stddev across queries: {all_rewards.std(axis=0, ddof=1).mean():.3}")

    return dict(output_path=results_dir)
