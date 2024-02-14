# %%
import os
from pprint import pprint
import random

import torch
import numpy as np
import wandb

from transformers import AutoTokenizer
from transformers import TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel, Trainer

max_val = 2
layers_to_use_ann = 3
layers_to_use_bob = 3
layers_to_use_cid = 1
# wandb.login()
# wandb.init(project="steg-hunt", name="diffie-hellman-hex-16:4-tr-12l:1l")

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "true"

ann = GPT2LMHeadModel.from_pretrained('gpt2').to("cuda")
bob = GPT2LMHeadModel.from_pretrained('gpt2').to("cuda")
cid = GPT2LMHeadModel.from_pretrained('gpt2').to("cuda")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

# use only the first few layers out of 12
ann._modules["transformer"]._modules["h"] = ann._modules["transformer"]._modules["h"][:layers_to_use_ann]
bob._modules["transformer"]._modules["h"] = ann._modules["transformer"]._modules["h"][:layers_to_use_bob]
cid._modules["transformer"]._modules["h"] = cid._modules["transformer"]._modules["h"][:layers_to_use_cid]

# %%

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, data):
        self.input_ids = data.to("cpu")
        self.attention_masks = np.ones_like(self.input_ids)
        assert len(self.input_ids.shape) == 2  # and self.input_ids.shape[1] <= 6

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
for model in (ann, bob, cid):
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            disable_tqdm=True,  # This disables the progress bars
            learning_rate=5e-4,
            num_train_epochs=1,
            # per_device_train_batch_size=16,
            gradient_accumulation_steps=1,
            dataloader_num_workers=2,
            optim="adamw_torch",
            output_dir="out",
            weight_decay=1e-2,
            report_to=[],
        ),
    )

    for i in range(2):
        logits = model(torch.tensor([2, 4, 10]).to("cuda")).logits
        eval = logits.argmax(axis=1)
        print(eval)

        data = np.random.randint(1, max_val + 1, size=(16, 3))
        data = torch.tensor(data)
        trainer.train_dataset = DatasetWrapper(data)
        trainer.train()


# %%
training_args = TrainingArguments(
    disable_tqdm=True,  # This disables the progress bars
    learning_rate=5e-4,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    # dataloader_num_workers=2,
    optim="adamw_torch",
    output_dir="out",
    weight_decay=1e-2,
    report_to=[],
)

ann_trainer = MyTrainer(
    model=ann,
    tokenizer=tokenizer,
    args=training_args,
)
bob_trainer = MyTrainer(
    model=bob,
    tokenizer=tokenizer,
    args=training_args,
)
cid_trainer = MyTrainer(
    model=cid,
    tokenizer=tokenizer,
    args=training_args,
)
# bob_trainer.args.learning_rate = 5e-4
# cid_trainer.args.learning_rate = 5e-4
# disable log printing
ann_trainer.log = lambda logs: None
bob_trainer.log = lambda logs: None
cid_trainer.log = lambda logs: None

# %%
gen_args = dict(
    max_new_tokens=1,
    pad_token_id=tokenizer.eos_token_id,
    temperature=0.5,
    do_sample=True,
)

good_token = 777
bad_token = 666
batch_size = 256 # training_args.per_device_train_batch_size
rand_size = batch_size // 4

# %%
while True:
# for i in range(1):
    # get random inputs
    g = torch.full((batch_size, 1), good_token, dtype=int).to("cuda")
    key = torch.randint(1, max_val + 1, size=(batch_size,1)).to("cuda")
    msg = torch.randint(1, max_val + 1, size=(batch_size,1)).to("cuda")

    # encrypt
    g_key_msg = torch.cat((g, key, msg), axis=1)
    g_key_msg_emsg = ann.generate(g_key_msg, **gen_args)
    emsg = g_key_msg_emsg[:, -1:]
    # emsg[:rand_size] = torch.randint(1, max_val + 1, size=(rand_size,1)).to("cuda")
    xored_ideal = ((key + msg) % 2) + 1
    emsg[:rand_size] = xored_ideal[:rand_size]

    # bob and cid try to decrypt
    key_emsg = torch.cat((key, emsg), axis=1)
    bmsg = bob.generate(key_emsg, **gen_args)[:, -1:]
    cmsg = cid.generate(emsg, **gen_args)[:, -1:] 

    # is it succesful
    success = (msg == bmsg).logical_and(msg != cmsg).squeeze()
    # success[:rand_size] = True
    failure = success.logical_not()

    # insert the first token to indicate whether transmission succeeded
    # todo? limit the number of bad examples 
    s_key_msg_emsg = g_key_msg_emsg.clone().detach()
    s_key_msg_emsg[failure, 0] = bad_token

    ### training ###

    # train alice
    # groups = []
    # for s in [good_token, bad_token]:
    #     s_mask = s_key_msg_emsg[:, 0] == s
    #     for val in range(1, max_val + 1):
    #         v_mask = s_key_msg_emsg[:, -1] == val
    #         sv_mask = s_mask.logical_and(v_mask)
    #         sv_indexes = sv_mask.nonzero().flatten()

    #         # shuffle
    #         perm = torch.randperm(len(sv_indexes))
    #         groups.append(sv_indexes[perm])

    # min_count = min(len(g) for g in groups)
    # equianswer_indexes = torch.cat([g[:min_count] for g in groups])
    # if len(equianswer_indexes) == 0:
    #     continue
    equianswer_indexes = list(range(len(s_key_msg_emsg)))

    ann_trainer.train_dataset = DatasetWrapper(s_key_msg_emsg[equianswer_indexes])
    ann_trainer.train()

    # train bob
    key_emsg_msg = torch.cat((key_emsg, msg), axis=1)
    bob_trainer.train_dataset = DatasetWrapper(key_emsg_msg[equianswer_indexes])
    bob_trainer.train()

    # train cid
    emsg_msg = torch.cat((emsg, msg), axis=1)
    cid_trainer.train_dataset = DatasetWrapper(emsg_msg[equianswer_indexes])
    cid_trainer.train()

    ### stats ###
    print("success, key, msg, emsg, bmsg, cmsg")
    print(torch.cat((s_key_msg_emsg, bmsg, cmsg), axis=1))
    stats = dict(
        # accuracy=float(success.float().mean()),
        accuracy=float(success[rand_size:].float().mean()),
        total_batch_len=len(equianswer_indexes),
    )
    pprint(stats)
    
# %%