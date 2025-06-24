"""
실험을 정의하고 실행하기 위한 유틸리티 함수들
(experiment launcher, job 구성, trial 파싱 등)
"""

from functools import partial, wraps
from typing import List, get_type_hints

from summarize_from_feedback.utils import jobs
from summarize_from_feedback.utils.combos import bind, combos

# 주로 main_fn(H: HParams)에서 HParams 타입을 추출할 때 사용됨
def get_annotation_of_only_argument(fn):
    # 함수 fn의 타입 힌트를 가져와서 
    annotations = get_type_hints(fn).values()
    
    # 인자가 하나인지 확인
    if len(annotations) != 1:
        raise ValueError(
            f"fn {fn} has {len(annotations)} arguments, but we wanted 1: {annotations}"
        )
    # 해당 인자의 타입 반환 (예: HParams)
    ty, = annotations
    return ty

# trial을 받아 실행 가능한 Job 객체 리스트로 변환
def get_experiment_jobs(name, launch_fn, trials, hparam_class=None) -> List[jobs.Job]:
    # hparam_class가 명시되지 않으면 launch_fn의 유일한 인자의 타입을 추론
    if hparam_class is None:
        hparam_class = get_annotation_of_only_argument(launch_fn)

    # Maps experiment def key to argument name for jobs.launch; these get pulled out
    # mpi, mode는 JobHParams 객체에 직접 매핑됨
    launch_kwarg_keys = dict(mpi="mpi", mode="mode")

    to_launch = []
    for trial in trials:
        descriptors = []
        trial_bindings = []
        trial_bindings_dict = {}  # only used for message

        # Extract bindings & descriptors from the trial
        # trial은 (key, value, metadata)의 튜플 리스트 
        for k, v, s in trial:
            if k is not None:
                if k in trial_bindings_dict:
                    print(f"NOTE: overriding key {k} from {trial_bindings_dict[k]} to {v}")
                trial_bindings.append((k, v))
                trial_bindings_dict[k] = v
            # descriptor는 실험 이름 구분에 사용됨 
            if "descriptor" in s and s["descriptor"] is not "":
                descriptors.append(str(s["descriptor"]))

        # Pull out arguments for jobs.launch
        # launch_H는 Job 메타데이터 (name, mpi, mode)를 저장
        # 나머지 파라미터는 filtered_trial_bindings에 따로 모음
        launch_H = jobs.JobHParams()
        launch_H.name = "/".join([name] + descriptors)
        filtered_trial_bindings = []
        dry_run = False
        for k, v in trial_bindings:
            if k in launch_kwarg_keys:
                setattr(launch_H, launch_kwarg_keys[k], v)
            elif k == "dry_run":
                dry_run = v
            else:
                filtered_trial_bindings.append((k, v))

        if dry_run:
            print(f"{launch_H.name}: {filtered_trial_bindings}")
        else:
            hparams = hparam_class()
            hparams.override_from_pairs(filtered_trial_bindings)
            to_launch.append(jobs.Job(fn=partial(launch_fn, hparams), params=launch_H))
    return to_launch

# experiment_dict에서 exp 키로 trial 목록을 찾고
def experiment_fn_launcher(experiment_dict, fn):
    def launcher(exp, name, **extra_args):
        try:
            trials = experiment_dict[exp]
        except KeyError:
            raise ValueError(f"Couldn't find experiment '{exp}'")
        # fn(name, trials, ...) 형식으로 실행 
        fn(name, trials, **extra_args)
    # 이 함수를 통해 fire가 CLI 인자를 받아 launcher를 실행할 수 있음 
    return launcher


def experiment_def_launcher(experiment_dict, main_fn, **default_bindings):
    """
    Use like this:

    if __name__ == "__main__":
        fire.Fire(
            experiment_def_launcher(
                experiment_dict=experiment_definitions(),
                main_fn=train_rm.main,
            )
        )
    """
    # experiment_dict와 실행 함수(main_fn)를 받아 Fire에서 쓸 수 있는 함수 생성
    @wraps(main_fn)
    def fn(name, trials, **extra_args):
        # Bind remaining extra arguments from the defaults and from the command line
        trials = combos(
            *[bind(k, v) for k, v in default_bindings.items()],
            trials,
            *[bind(k, v) for k, v in extra_args.items()],
        )
        # get_experiment_jobs(...)로 Job 리스트 생성 
        exp_jobs = get_experiment_jobs(name, main_fn, trials)
        # jobs.multilaunch(...)로 모든 Job 실행 
        return jobs.multilaunch(exp_jobs)

    return experiment_fn_launcher(experiment_dict, fn)
