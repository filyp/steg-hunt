# %%
import os
import random

import torch
import numpy as np

from trl import SFTTrainer
from transformers import TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel, Trainer

max_val = 11
layers_to_use = 3
os.environ["TOKENIZERS_PARALLELISM"] = "true"

model = GPT2LMHeadModel.from_pretrained('gpt2').to("cuda")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

# %%
# use only the first few layers out of 12
model._modules["transformer"]._modules["h"] = model._modules["transformer"]._modules["h"][:layers_to_use]

# %%

np.random.randint(1, max_val + 1, size=(20, 3))
# %%
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_size):
        self.input_ids = np.random.randint(1, max_val + 1, size=(dataset_size, 3))
        self.attention_masks = np.ones_like(self.input_ids)
        # self.labels = np.random.randint(1, max_val + 1, size=(dataset_size, 3))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],  # task and reasoning, joined and padded
            attention_mask=self.attention_masks[i],
            # labels=self.labels[i],
        ) 


# class MyTrainer(SFTTrainer):
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

trainer = MyTrainer(
    model=model,
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
# disable log printing
# trainer.log = lambda logs: None


# %%
for i in range(5):
    logits = model(torch.tensor([2, 4, 10]).to("cuda")).logits
    eval = logits.argmax(axis=1)
    print(eval)

    trainer.train_dataset = MyDataset(64)
    trainer.train()

# %%
# t2 = SFTTrainer(
#     model=model,
#     dataset_text_field="text",
# )

# t2.save_model("../model_l3")
# output_dir = "../models"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
# self.tokenizer.save_pretrained(output_dir)
