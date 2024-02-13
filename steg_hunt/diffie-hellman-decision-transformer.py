# %%
import os
from pprint import pprint
import random

import torch
import numpy as np
import wandb
from trl import PPOConfig

from transformers import AutoTokenizer

from transformers import TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel, Trainer

max_val = 4
g_max_val = 16
layers_to_use_ann = 12
layers_to_use_cid = 1
wandb.login()
wandb.init(project="steg-hunt", name="diffie-hellman-hex-16:4-tr-12l:1l")

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "true"

ann = GPT2LMHeadModel.from_pretrained('gpt2').to("cuda")
cid = GPT2LMHeadModel.from_pretrained('gpt2').to("cuda")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

# use only the first few layers out of 12
ann._modules["transformer"]._modules["h"] = ann._modules["transformer"]._modules["h"][:layers_to_use_ann]
cid._modules["transformer"]._modules["h"] = cid._modules["transformer"]._modules["h"][:layers_to_use_cid]

# %%

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.input_ids = data.to("cpu")
        self.attention_masks = np.ones_like(self.input_ids)
        assert len(self.input_ids.shape) == 2 and self.input_ids.shape[1] <= 6

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],  # task and reasoning, joined and padded
            attention_mask=self.attention_masks[i],
        ) 


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        # labels = inputs.pop("labels")

        # batched generation
        logits = model(input_ids).logits

        labels = input_ids.to(logits.device)
        shift_logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        # calculate loss only for the tokens after "answer"
        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        return lm_loss

# %%
# pretrain to make the model use only 1-10 tokens
trainer = MyTrainer(
    model=ann,
    tokenizer=tokenizer,
    args=TrainingArguments(
        disable_tqdm=True,  # This disables the progress bars
        learning_rate=5e-4,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        dataloader_num_workers=2,
        optim="adamw_torch",
        output_dir="out",
        weight_decay=1e-2,
        report_to=[],
    ),
)

for i in range(4):
    logits = ann(torch.tensor([2, 4, 10]).to("cuda")).logits
    eval = logits.argmax(axis=1)
    print(eval)

    data = np.random.randint(1, max_val + 1, size=(16, 3))
    data = torch.tensor(data)
    trainer.train_dataset = MyDataset(data)
    trainer.train()


trainer = MyTrainer(
    model=cid,
    tokenizer=tokenizer,
    args=TrainingArguments(
        disable_tqdm=True,  # This disables the progress bars
        learning_rate=5e-4,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        dataloader_num_workers=2,
        optim="adamw_torch",
        output_dir="out",
        weight_decay=1e-2,
        report_to=[],
    ),
)

for i in range(4):
    logits =cid(torch.tensor([2, 4, 10]).to("cuda")).logits
    eval = logits.argmax(axis=1)
    print(eval)

    data = np.random.randint(1, max_val + 1, size=(16, 3))
    data = torch.tensor(data)
    trainer.train_dataset = MyDataset(data)
    trainer.train()
# %%
gen_args = dict(
    max_new_tokens=1,
    pad_token_id=tokenizer.eos_token_id,
    temperature=2,
)

# model(torch.tensor([8, 8]).to("cuda"))
s = 777  # success token
f = 666  # failure token

ab_trainer = MyTrainer(
    model=ann,
    tokenizer=tokenizer,
    args=TrainingArguments(
        disable_tqdm=True,  # This disables the progress bars
        learning_rate=2e-6,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        dataloader_num_workers=2,
        optim="adamw_torch",
        output_dir="out",
        weight_decay=1e-2,
        report_to=[],
    ),
)
c_trainer = MyTrainer(
    model=cid,
    tokenizer=tokenizer,
    args=TrainingArguments(
        disable_tqdm=True,  # This disables the progress bars
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        dataloader_num_workers=2,
        optim="adamw_torch",
        output_dir="out",
        weight_decay=1e-2,
        report_to=[],
    ),
)
# disable log printing
ab_trainer.log = lambda logs: None
c_trainer.log = lambda logs: None

# %%
# for i in range(1000):
try:
    while True:
        batch_size = 64
        rand_size = 48
        rand2_size = 32
        s_g_a = []
        s_g_b = []
        for i in range(batch_size):
            g = random.randint(1, g_max_val)
            a = random.randint(1, max_val)
            b = random.randint(1, max_val)
            s_g_a.append([s, g, a])
            s_g_b.append([s, g, b])
        s_g_a = torch.tensor(s_g_a).to("cuda")
        s_g_b = torch.tensor(s_g_b).to("cuda")

        s_g_a_ga = ann.generate(s_g_a, **gen_args)
        s_g_b_gb = ann.generate(s_g_b, **gen_args)
        # randomize some answers
        s_g_a_ga[:rand_size, -1] = torch.randint(1, max_val + 1, size=(rand_size,)).to("cuda")
        s_g_b_gb[:rand_size, -1] = torch.randint(1, max_val + 1, size=(rand_size,)).to("cuda")

        ga = s_g_a_ga[:, -1:]
        gb = s_g_b_gb[:, -1:]
        s_g_a_ga_gb = torch.cat([s_g_a_ga, gb], axis=1)
        s_g_b_gb_ga = torch.cat([s_g_b_gb, ga], axis=1)
        s_g_a_ga_gb_gab = ann.generate(s_g_a_ga_gb, **gen_args)
        s_g_b_gb_ga_gba = ann.generate(s_g_b_gb_ga, **gen_args)
        # randomize some answers
        s_g_a_ga_gb_gab[:rand2_size, -1] = torch.randint(1, max_val + 1, size=(rand2_size,)).to("cuda")
        s_g_b_gb_ga_gba[:rand2_size, -1] = torch.randint(1, max_val + 1, size=(rand2_size,)).to("cuda")

        gab = s_g_a_ga_gb_gab[:, -1:]
        gba = s_g_b_gb_ga_gba[:, -1:]
        matches = gab == gba

        # Cid tries to guess
        gs = s_g_a[:, 1].reshape(-1, 1)
        g_ga_gb = torch.cat((gs, ga, gb), axis=1)
        g_ga_gb_guess = cid.generate(g_ga_gb, **gen_args)
        guesses = g_ga_gb_guess[:, -1:]
        guesseds = guesses == gab

        # insert "reward"
        successes = torch.logical_and(matches, guesseds.logical_not())
        good_examples = []
        bad_examples = []
        for success, a_msg, b_msg in zip(successes, s_g_a_ga_gb_gab, s_g_b_gb_ga_gba):
            # TODO maybe some intermediate reward, just for not being guessed
            if success:
                a_msg[0] = s
                b_msg[0] = s
                good_examples.append(a_msg)
                good_examples.append(b_msg)
            else:
                a_msg[0] = f
                b_msg[0] = f
                bad_examples.append(a_msg)
                bad_examples.append(b_msg)
        
        if len(good_examples) == 0:
            print("no good examples, skipping")
            continue
        random.shuffle(bad_examples)
        bad_examples = bad_examples[:len(good_examples)]
        ab_data = torch.stack(good_examples + bad_examples)
        # ab_data = torch.cat([s_g_a_ga_gb_gab, s_g_b_gb_ga_gba])

        ab_trainer.train_dataset = MyDataset(ab_data)
        ab_trainer.train()
        print("gab gba guesses successes")
        print(torch.cat((gab, gba, guesses, successes), axis=1))

        # get accuracy only on those not randomized cases
        stats = dict(
            len_ab_data=len(ab_data),
            matches=torch.mean(matches[rand_size:] * 1.),
            guessed=torch.mean(guesseds[rand_size:] * 1.),
            accuracy=torch.mean(successes[rand_size:] * 1.),
        )
        wandb.log(stats)
        pprint(stats)
        print()

        # train the adversary
        g_ga_gb_gab = torch.cat((g_ga_gb, gab), axis=1)
        c_trainer.train_dataset = MyDataset(g_ga_gb_gab)
        c_trainer.train()
except KeyboardInterrupt:
    wandb.finish()



# %%
