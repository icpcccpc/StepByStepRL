# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.model import compute_position_id_with_mask
from tensordict import TensorDict


class MyRayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def compute_prompt_response_length(self, data: DataProto):
        prompt_mask = data.batch["attention_mask"][
            :, : -self.config.data.max_response_length
        ]
        response_mask = data.batch["attention_mask"][
            :, -self.config.data.max_response_length :
        ]
        prompt_length = prompt_mask.sum(dim=1)
<<<<<<< HEAD
        response_length = response_mask.sum(dim=1).to(torch.float32)
=======
        response_length = response_mask.sum(dim=1)
>>>>>>> 2d6a3de (add_data)
        return prompt_length, response_length

    def compute_response_prefix_suffix_length(
        self, prompt_length, response_length, n_samples
    ):
        """random split"""
        assert (
            response_length.shape[0] % n_samples == 0
        ), f"can't divide when computing split length , {response_length.shape[0]} % {n_samples} != 0"
<<<<<<< HEAD
        grouped_response_length = response_length.reshape(-1, n_samples)
        grouped_response_length_min = grouped_response_length.min(dim=1).values
        grouped_response_length_mean = grouped_response_length.mean(dim=1)
=======
        response_length = response_length.to(torch.float32)
        grouped_response_length = response_length.reshape(-1, n_samples)
        grouped_response_length_mean = grouped_response_length.mean(dim=1)
        grouped_response_length_min = grouped_response_length.min(dim=1).values
>>>>>>> 2d6a3de (add_data)
        grouped_response_length_ratio = 0.3 + 0.4 * torch.rand(
            grouped_response_length_mean.shape,
            device=grouped_response_length_mean.device,
        )
        # 0.3 ~ 0.7
        split_index1 = (grouped_response_length_min * 0.8).int()
        split_index2 = (
            grouped_response_length_mean * grouped_response_length_ratio
        ).int()
        split_index = torch.minimum(split_index1, split_index2)
        split_index = torch.repeat_interleave(split_index, n_samples)
        split_index = torch.minimum(
            split_index, self.config.data.max_prompt_length - prompt_length
        )  # don't truncate the prompt
        return split_index.cpu().numpy()

    # default padding left
    def split_to_3parts(self, data: DataProto, n_samples: int) -> dict:
        """non_tensorbatch : prompt_ids response_prefix_ids response_suffix_ids"""

        prompt_ids = data.batch["input_ids"][:, : -self.config.data.max_response_length]
        response_ids = data.batch["input_ids"][
            :, -self.config.data.max_response_length :
        ]
        prompt_length, response_length = self.compute_prompt_response_length(data)
        split_index = self.compute_response_prefix_suffix_length(
            prompt_length, response_length, n_samples
        )
        batch_size = len(data)
        prompt_ids_lst = []
        response_prefix_ids_lst = []
<<<<<<< HEAD
        for i in range(batch_size):
            prompt_ids_lst.append(prompt_ids[i, -prompt_length[i] :])
            response_prefix_ids_lst.append(response_ids[i, : split_index[i] + 1])
        return {
            "prompts": prompt_ids_lst,
            "response_prefixes": response_prefix_ids_lst,
=======
        response_suffix_ids_lst = []
        for i in range(batch_size):
            prompt_ids_lst.append(prompt_ids[i, -prompt_length[i] :])
            response_prefix_ids_lst.append(response_ids[i, : split_index[i] + 1])
            response_suffix_ids_lst.append(
                response_ids[i, split_index[i] + 1 : response_length[i]]
            )
        return {
            "prompts": prompt_ids_lst,
            "response_prefixes": response_prefix_ids_lst,
            "response_suffixes": response_suffix_ids_lst,
>>>>>>> 2d6a3de (add_data)
            "split_index": split_index,
        }

    def prompts_and_response_prefix_to_gen_batch(self, prompts, response_prefixes):
        """get gen_batch for generation"""
        batch_size = len(prompts)
        length_lst = [
            len(prompt) + len(response_prefix)
            for prompt, response_prefix in zip(prompts, response_prefixes)
        ]
        max_length = max(length_lst)
        assert (
            max_length <= self.config.data.max_prompt_length
        ), "func prompts_and_response_prefix_to_gen_batch , max_length too long"
        max_length = self.config.data.max_prompt_length

        input_ids = torch.full(
            size=(batch_size, max_length),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=prompts[0].device,
        )
        attention_mask = torch.zeros(
            size=(batch_size, max_length), dtype=torch.long, device=prompts[0].device
        )
        for i, (prompt, response_prefix) in enumerate(zip(prompts, response_prefixes)):
            prompt_length = len(prompt)
            response_prefix_length = len(response_prefix)
            total_length = prompt_length + response_prefix_length
            input_ids[i, -total_length:-response_prefix_length] = prompt
            input_ids[i, -response_prefix_length:] = response_prefix
            attention_mask[i, -total_length:] = 1
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        return DataProto.from_dict(tensors=batch_dict)

    def gen_batch_and_response_to_output_DataProto(
        self, gen_batch, response_lst, n_samples=1
    ):
        """uniform to the same format
        keys:prompts responses input_ids(whole seq) attention_mask position_ids"""

        prompts_tensor = gen_batch.batch["input_ids"]
        prompts_attention_mask = gen_batch.batch["attention_mask"]

        assert prompts_tensor.shape[0] == len(
            response_lst
        ), f" prompts_response_to_output_DataProto size wrong , prompts_tensor {prompts_tensor.shape[0]} != response_lst {len(response_lst)}"
        batch_size = prompts_tensor.shape[0]
        prompts_length = prompts_tensor.shape[1]
        assert (
            prompts_length == self.config.data.max_prompt_length
        ), "gen_batch_and_response_to_output_DataProto , prompts_length wrong"
        response_length = self.config.data.max_response_length
        input_ids = torch.full(
            size=(batch_size, prompts_length + response_length),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=prompts_tensor.device,
        )
        attention_mask = torch.zeros(
            size=(batch_size, prompts_length + response_length),
            dtype=torch.long,
            device=prompts_attention_mask.device,
        )
        attention_mask[:, :prompts_length] = prompts_attention_mask
        input_ids[:, :prompts_length] = prompts_tensor
        for i, response in enumerate(response_lst):
            input_ids[i, prompts_length : prompts_length + len(response)] = response
            attention_mask[i, prompts_length : prompts_length + len(response)] = 1
        position_ids = compute_position_id_with_mask(attention_mask)
        batch = TensorDict(
            {
                "prompts": input_ids[:, :prompts_length],
                "responses": input_ids[:, prompts_length:],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response_mask": attention_mask[
                    :, prompts_length:
                ],  # don't have <eos>.compute response mask manually.
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch)

<<<<<<< HEAD
=======
    def batch_concat_ratio(self, batch1, batch2, ratio):
        """batch1 = batch2 * ratio && interleave"""
        assert (
            len(batch1) == len(batch2) * ratio
        ), f" {len(batch1)=} != {len(batch2)*ratio=}"

        batch_merged = DataProto.concat([batch1, batch2])

        reorder_indices = []
        for i in range(len(batch2)):
            reorder_indices.append(len(batch1) + i)
            reorder_indices.extend(list(range(i * ratio, (i + 1) * ratio)))
        reorder_indices = torch.tensor(reorder_indices, dtype=torch.int)
        batch_merged.reorder(reorder_indices)
        return batch_merged

>>>>>>> 2d6a3de (add_data)
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.start_profile(
                            role="e2e", profile_step=self.global_steps
                        )
                        if self.use_reference_policy:
                            self.ref_policy_wg.start_profile()
                        if self.use_critic:
                            self.critic_wg.start_profile()
                        if self.use_rm:
                            self.rm_wg.start_profile()

                new_batch_layer1: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch_layer1.non_tensor_batch.keys():
                    raise NotImplementedError
                    gen_batch_layer1 = new_batch_layer1.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch_layer1 = new_batch_layer1.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch_layer1 = gen_batch_layer1.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n_layer1,
                    interleave=True,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            gen_batch_layer1
                        )
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        raise NotImplementedError
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = (
                                self.actor_rollout_wg.generate_sequences(
                                    gen_baseline_batch
                                )
                            )

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(
                                batch_keys=list(gen_baseline_output.batch.keys())
                            )

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
<<<<<<< HEAD

                    new_batch_layer1.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch_layer1.batch))],
                        dtype=object,
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch_layer1 = new_batch_layer1.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n_layer1,
                        interleave=True,
                    )
                    # batch = batch.union(gen_batch_output)
                    prompts_response_prefix_suffix_dict = self.split_to_3parts(
                        gen_batch_output, self.config.actor_rollout_ref.rollout.n_layer1
                    )
                    # prompts | response_prefixes | response_suffixes
                    gen_batch_layer2 = self.prompts_and_response_prefix_to_gen_batch(
                        prompts=prompts_response_prefix_suffix_dict["prompts"],
                        response_prefixes=prompts_response_prefix_suffix_dict[
                            "response_prefixes"
                        ],
                    )
                    gen_batch_layer2 = gen_batch_layer2.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n_layer2,
                        interleave=True,
                    )
                    with marked_timer("gen_layer2", timing_raw, "red"):
                        gen_batch_layer2_output = (
                            self.actor_rollout_wg.generate_sequences(gen_batch_layer2)
                        )
                        timing_raw.update(gen_batch_layer2_output.meta_info["timing"])
                        gen_batch_layer2_output.meta_info.pop("timing", None)

                    new_batch_layer1.non_tensor_batch["layer1_uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch_layer1.batch))],
                        dtype=object,
                    )
                    new_batch_layer1.non_tensor_batch["split_index"] = np.array(
                        prompts_response_prefix_suffix_dict["split_index"], dtype=int
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch_layer2 = new_batch_layer1.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n_layer2,
                        interleave=True,
                    )
                    new_batch_layer2 = new_batch_layer2.union(gen_batch_layer2_output)
                    if "response_mask" not in new_batch_layer2.batch:
                        new_batch_layer2.batch["response_mask"] = compute_response_mask(
                            new_batch_layer2
                        )

                    new_batch_layer1 = new_batch_layer1.union(
                        self.gen_batch_and_response_to_output_DataProto(
                            gen_batch=gen_batch_layer1,
                            response_lst=prompts_response_prefix_suffix_dict[
                                "response_prefixes"
                            ],
                        )
                    )
                    assert "response_mask" in new_batch_layer1.batch
=======
                    with marked_timer("process_data", timing_raw, "red"):
                        new_batch_layer1.non_tensor_batch["uid"] = np.array(
                            [
                                str(uuid.uuid4())
                                for _ in range(len(new_batch_layer1.batch))
                            ],
                            dtype=object,
                        )
                        # repeat to align with repeated responses in rollout
                        new_batch_layer1 = new_batch_layer1.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n_layer1,
                            interleave=True,
                        )
                        if self.config.algorithm.filter_groups.enable:
                            with marked_timer("reward filter", timing_raw, "red"):
                                new_batch_layer1 = new_batch_layer1.union(
                                    gen_batch_output
                                )
                                try:
                                    reward_result = self.reward_fn(
                                        new_batch_layer1, return_dict=True
                                    )
                                    reward_tensor = reward_result["reward_tensor"]
                                except Exception as e:
                                    print(f"Error in reward_fn: {e}")
                                    reward_tensor = self.reward_fn(new_batch)
                                layer_scores = reward_tensor.sum(dim=-1).reshape(
                                    -1, self.config.actor_rollout_ref.rollout.n_layer1
                                )
                                layer_mask = (
                                    torch.repeat_interleave(
                                        layer_scores.std(dim=-1) > 0,
                                        self.config.actor_rollout_ref.rollout.n_layer1,
                                    )
                                    .cpu()
                                    .numpy()
                                )
                                kept_rate = layer_mask.sum() / len(layer_mask)
                                print(f"{kept_rate=}")
                                kept_ids = np.nonzero(layer_mask)[0].tolist()
                                # self.consistent_select_idxs( new_batch_layer1,kept_ids )
                                new_batch_layer1 = new_batch_layer1[kept_ids]
                                gen_batch_layer1 = gen_batch_layer1[kept_ids]
                                gen_batch_output = new_batch_layer1.pop(
                                    batch_keys=[
                                        "input_ids",
                                        "attention_mask",
                                        "position_ids",
                                    ]
                                )
                                new_batch_layer1.pop(
                                    batch_keys=["prompts", "responses"]
                                )
                        # batch = batch.union(gen_batch_output)
                        prompts_response_prefix_suffix_dict = self.split_to_3parts(
                            gen_batch_output,
                            self.config.actor_rollout_ref.rollout.n_layer1,
                        )
                        # prompts | response_prefixes | response_suffixes
                        gen_batch_layer2 = (
                            self.prompts_and_response_prefix_to_gen_batch(
                                prompts=prompts_response_prefix_suffix_dict["prompts"],
                                response_prefixes=prompts_response_prefix_suffix_dict[
                                    "response_prefixes"
                                ],
                            )
                        )

                        new_batch_layer1.non_tensor_batch["layer1_uid"] = np.array(
                            [
                                str(uuid.uuid4())
                                for _ in range(len(new_batch_layer1.batch))
                            ],
                            dtype=object,
                        )
                        new_batch_layer1.non_tensor_batch["split_index"] = np.array(
                            prompts_response_prefix_suffix_dict["split_index"],
                            dtype=int,
                        )

                        new_batch_layer2_part0 = new_batch_layer1.repeat(repeat_times=1)
                        new_batch_layer2_part0 = new_batch_layer2_part0.union(
                            self.gen_batch_and_response_to_output_DataProto(
                                gen_batch=gen_batch_layer2,
                                response_lst=prompts_response_prefix_suffix_dict[
                                    "response_suffixes"
                                ],
                            )
                        )

                        gen_batch_layer2 = gen_batch_layer2.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n_layer2,
                            interleave=True,
                        )
                        with marked_timer("gen_layer2", timing_raw, "red"):
                            gen_batch_layer2_output = (
                                self.actor_rollout_wg.generate_sequences(
                                    gen_batch_layer2
                                )
                            )
                            timing_raw.update(
                                gen_batch_layer2_output.meta_info["timing"]
                            )
                            gen_batch_layer2_output.meta_info.pop("timing", None)
                        # repeat to align with repeated responses in rollout

                        new_batch_layer2_part1 = new_batch_layer1.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n_layer2,
                            interleave=True,
                        )
                        new_batch_layer2_part1 = new_batch_layer2_part1.union(
                            gen_batch_layer2_output
                        )
                        if "response_mask" not in new_batch_layer2_part1.batch:
                            new_batch_layer2_part1.batch["response_mask"] = (
                                compute_response_mask(new_batch_layer2_part1)
                            )
                        # new_batch_layer2 = DataProto.concat( [ new_batch_layer2_part0,new_batch_layer2_part1 ])
                        new_batch_layer2 = self.batch_concat_ratio(
                            new_batch_layer2_part1,
                            new_batch_layer2_part0,
                            self.config.actor_rollout_ref.rollout.n_layer2,
                        )
                        new_batch_layer1 = new_batch_layer1.union(
                            self.gen_batch_and_response_to_output_DataProto(
                                gen_batch=gen_batch_layer1,
                                response_lst=prompts_response_prefix_suffix_dict[
                                    "response_prefixes"
                                ],
                            )
                        )
                        assert "response_mask" in new_batch_layer1.batch
>>>>>>> 2d6a3de (add_data)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            raise NotImplementedError
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(
                                new_batch_layer2, return_dict=True
                            )
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get(
                                "reward_extra_info", {}
                            )
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}
                        # TODO: compute the average rewards of the same uid_prefix as the rewards of new_batch_layer1
                        new_batch_layer2.batch["token_level_scores"] = reward_tensor
                        print(f"{list(reward_extra_infos_dict.keys())=}")

                        if reward_extra_infos_dict:
                            new_batch_layer2.non_tensor_batch.update(
                                {
                                    k: np.array(v)
                                    for k, v in reward_extra_infos_dict.items()
                                }
                            )

                        prompt_prefix_uid2_scores = defaultdict(list)
                        layer_scores = (
                            new_batch_layer2.batch["token_level_scores"]
                            .sum(dim=-1)
                            .numpy()
                        )
                        for prefix_uid, score in zip(
                            new_batch_layer2.non_tensor_batch["layer1_uid"],
                            layer_scores,
                        ):
                            prompt_prefix_uid2_scores[prefix_uid].append(score)
                        layer1_batch_size = len(new_batch_layer1)
                        new_batch_layer1.batch["token_level_scores"] = torch.zeros(
                            size=(
                                layer1_batch_size,
                                self.config.data.max_response_length,
                            ),
                            dtype=new_batch_layer2.batch["token_level_scores"].dtype,
                            device=new_batch_layer1.batch.device,
                        )
                        for i, (split_index, uid) in enumerate(
                            zip(
                                new_batch_layer1.non_tensor_batch["split_index"],
                                new_batch_layer1.non_tensor_batch["layer1_uid"],
                            )
                        ):
                            new_batch_layer1.batch["token_level_scores"][
                                i, split_index
                            ] = float(np.mean(prompt_prefix_uid2_scores[uid]))

                        if reward_extra_infos_dict:
                            # for k,v in reward_extra_infos_dict.items():
                            #    print(k,v)
                            new_batch_layer1.non_tensor_batch.update(
                                {
                                    k: np.zeros(
                                        shape=(layer1_batch_size,),
                                        dtype=new_batch_layer2.non_tensor_batch[
                                            k
                                        ].dtype,
                                    )
                                    for k in reward_extra_infos_dict.keys()
                                }
                            )

                        # keys need to be the same
                        assert (
                            new_batch_layer1.batch.keys()
                            == new_batch_layer2.batch.keys()
                        ), f"{new_batch_layer1.batch.keys()} != {new_batch_layer2.batch.keys()}"
                        assert (
                            new_batch_layer1.non_tensor_batch.keys()
                            == new_batch_layer2.non_tensor_batch.keys()
                        ), f"{new_batch_layer1.non_tensor_batch.keys()} != {new_batch_layer2.non_tensor_batch.keys()}"
                        # for k in new_batch_layer1.batch.keys():
                        #    print(k , new_batch_layer1.batch[k].shape ,new_batch_layer2.batch[k].shape  )
                        # for k in new_batch_layer1.non_tensor_batch.keys():
                        #    print(k , new_batch_layer1.non_tensor_batch[k].shape ,new_batch_layer2.non_tensor_batch[k].shape  )

                        new_batch_layer2.non_tensor_batch["uid"] = (
                            new_batch_layer2.non_tensor_batch["layer1_uid"]
                        )
<<<<<<< HEAD
                        new_batch_layer1_list = [
                            new_batch_layer1[
                                i : i + self.config.actor_rollout_ref.rollout.n_layer1
                            ]
                            for i in range(
                                0,
                                len(new_batch_layer1),
                                self.config.actor_rollout_ref.rollout.n_layer1,
                            )
                        ]
                        rollout_n = (
                            self.config.actor_rollout_ref.rollout.n_layer1
                            * self.config.actor_rollout_ref.rollout.n_layer2
                        )
                        new_batch_layer2_list = [
                            new_batch_layer2[i : i + rollout_n]
                            for i in range(0, len(new_batch_layer2), rollout_n)
                        ]
                        new_batch_list = []

                        for mini_batch_layer1, mini_batch_layer2 in zip(
                            new_batch_layer1_list, new_batch_layer2_list
                        ):
                            new_batch_list.append(
                                DataProto.concat([mini_batch_layer1, mini_batch_layer2])
                            )
                        new_batch = DataProto.concat(new_batch_list)
=======
                        # new_batch = DataProto.concat( [new_batch_layer1 , new_batch_layer2])
                        new_batch = self.batch_concat_ratio(
                            new_batch_layer2,
                            new_batch_layer1,
                            self.config.actor_rollout_ref.rollout.n_layer2 + 1,
                        )

>>>>>>> 2d6a3de (add_data)
                        """----------------------------------------------------------------"""

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch[
                                "token_level_scores"
                            ]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"]
                                .sum(dim=-1)
                                .numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"]
                                .sum(dim=-1)
                                .numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"],
                            new_batch.non_tensor_batch[metric_name],
                            strict=True,
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
<<<<<<< HEAD
                        num_prompt_in_batch += len(kept_prompt_uids)
=======
>>>>>>> 2d6a3de (add_data)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(
                            new_batch.non_tensor_batch["uid"]
                        ):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
<<<<<<< HEAD

=======
                        kept_rate2 = len(kept_traj_idxs) / len(new_batch)
                        print(f"{kept_rate2=}")
>>>>>>> 2d6a3de (add_data)
                        new_batch = new_batch[kept_traj_idxs]
                        batch = (
                            new_batch
                            if batch is None
                            else DataProto.concat([batch, new_batch])
                        )
<<<<<<< HEAD
=======
                        num_prompt_in_batch = len(batch)
>>>>>>> 2d6a3de (add_data)

                        prompt_bsz = self.config.data.train_batch_size
                        if (
                            num_prompt_in_batch
                            < prompt_bsz
                            * self.config.actor_rollout_ref.rollout.n_layer1
<<<<<<< HEAD
                        ):
                            print(
                                f"{num_prompt_in_batch=} < {prompt_bsz*self.config.actor_rollout_ref.rollout.n_layer1=}"
=======
                            * self.config.actor_rollout_ref.rollout.n_layer2
                        ):
                            print(
                                f"{num_prompt_in_batch=} < {prompt_bsz*self.config.actor_rollout_ref.rollout.n_layer1*self.config.actor_rollout_ref.rollout.n_layer2=}"
>>>>>>> 2d6a3de (add_data)
                            )
                            max_num_gen_batches = (
                                self.config.algorithm.filter_groups.max_num_gen_batches
                            )
                            if (
                                max_num_gen_batches <= 0
                                or num_gen_batches < max_num_gen_batches
                            ):
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = (
                                self.config.data.train_batch_size
                                * self.config.actor_rollout_ref.rollout.n_layer1
                                * self.config.actor_rollout_ref.rollout.n_layer2
                            )
                            batch = batch[:traj_bsz]

                    # === Updating ===
                    assert "response_mask" in batch.batch.keys()
                    """DELETED batch.batch["response_mask"] = compute_response_mask(batch)"""

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = (
                            self.config.actor_rollout_ref.actor.loss_agg_mode
                        )
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item()
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n_layer2,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
<<<<<<< HEAD
=======

>>>>>>> 2d6a3de (add_data)
