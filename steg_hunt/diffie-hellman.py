# %%
import random

import torch
import numpy as np
import wandb
from trl import PPOConfig

from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl import PPOTrainer

max_val = 10
layers_to_use = 3
# wandb.login()
# wandb.init(project="steg-hunt", name="gpt2-3l-onlyRL-dummy")

# %%
config = PPOConfig(
    # model_name="../m2",
    model_name="../model_l3",
    learning_rate=1.4e-5,
    batch_size=32,
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ann = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
# bob = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
# cid = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
# bob = ann
# # use only the first few layers out of 12
ann._modules["pretrained_model"].transformer.h = ann._modules["pretrained_model"].transformer.h[:layers_to_use]
# bob._modules["pretrained_model"].transformer.h = bob._modules["pretrained_model"].transformer.h[:layers_to_use]


# %%
ann_trainer = PPOTrainer(
    model=ann,
    config=config,
    tokenizer=tokenizer,
)
# bob_trainer = PPOTrainer(
#     model=bob,
#     config=config,
#     tokenizer=tokenizer,
# )
# cid_trainer = PPOTrainer(
#     model=cid,
#     config=config,
#     tokenizer=tokenizer,
# )

# %%
generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
# %%
# this cell is one "epoch"
total_examples = 0
try:
    while True:
        g_as = []
        g_bs = []
        for i in range(config.batch_size):
            g = random.randint(1, max_val)
            a = random.randint(1, max_val)
            b = random.randint(1, max_val)
            g_a = torch.tensor([g, a]).to("cuda")
            g_b = torch.tensor([g, b]).to("cuda")
            g_as.append(g_a)
            g_bs.append(g_b)

        g_a_gas = ann_trainer.generate(g_as, max_length=3, **generation_kwargs)
        # g_b_gbs = bob_trainer.generate(g_bs, max_length=3, **generation_kwargs)
        g_b_gbs = g_a_gas

        ann_rewards = []
        bob_rewards = []
        for g_a_ga, g_b_gb in zip(g_a_gas, g_b_gbs):
            ga = g_a_ga[-1]
            gb = g_b_gb[-1]
            ann_reward = (ga == 3) * 1.0
            bob_reward = (ga == gb) * 1.0
            # print(ga, ann_reward)
            ann_reward += (1 <= ga <= max_val) * 0.1 - 0.1
            bob_reward += (1 <= gb <= max_val) * 0.1 - 0.1
            ann_rewards.append(ann_reward)
            bob_rewards.append(bob_reward)

        ann_trainer.step(g_as, g_a_gas, ann_rewards)
        # bob_trainer.step(g_bs, g_b_gbs, bob_rewards)

        # for g_a_ga, g_b_gb in zip(g_a_gas[:20], g_b_gbs[:20]):
        #     ga = g_a_ga[-1]
        #     gb = g_b_gb[-1]
        #     print(f"{int(ga):5}  {int(gb):5}")
        
        
        total_examples += len(ann_rewards)
        results = dict(
            total_examples=total_examples,
            avg_ann_reward=torch.stack(ann_rewards).mean().to("cpu"),
            # avg_bob_reward=torch.stack(bob_rewards).mean().to("cpu"),
        )
        # wandb.log(results)
        print(results["avg_ann_reward"])


except KeyboardInterrupt:
    # wandb.finish()
    pass


# anns_gab = ann.generate([g, a, gb])
# bobs_gab = bob.generate([g, b, ga])
# ann_bob_reward = (anns_gab == bobs_gab) * matching_answer_weight
# ann_trainer.step([g, a], [])
# %%
# ann(torch.tensor([7,8]).to("cuda"))[0][1][:12]
ann_rewards