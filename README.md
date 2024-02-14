# steg-hunt

Toy experiments for researching steganography with LLMs

# Usage

To quickly setup a docker:

```bash
cd /workspace && git clone https://github.com/filyp/steg-hunt.git && cd steg-hunt && pip install -r requirements.txt
```

# Notes

The problem with the current multi-agent decision transformers setup, is that when Alice tries some new encoding, which could have maybe worked eventually, Bob at first doesn't understand that encoding. So then Alice is told that this didn't work and she learns not to do it, before Bob and Cid even have time to adapt. Eventually, Alice's policy becomes something very rigid, even if we add some random actions to what she does. (Btw, that is a kind of stag hunt problem between Alice and Bob, but I realized that only after naming the repo steg-hunt lol)

So the problem is that to know whether a change to Alice's policy is good, we'd need to look into the future and see how it would work after Bob and Cid have already been trained some time with Alice using that new policy. On first sight, it seems computationally horrible.

**Maybe there is some insight from RL or MARL that helps deal with this.**

In send_secret.py, Alice was able to learn to XOR the one-bit message with the key, but only after some of the training examples demonstrated that XOR behavior. She was able to do that even if they weren't labeled as correct yet (because Bob didn't understand many of them). Only later Bob learned, and they became correct. As little as 1/16 of training data set to showcase XOR policy was enough to kickstart the grokking. (If we balance the answers for the training data. If we don't balance, 1/8 was needed. Or if we don't balance, but we use setup from send_secret_a_waits.py, we need only 1/12, but the results are quite noisy.)

(What seems to help, is training for several steps on the same evaluatios of encoding success. I'm not sure though.)

Takeaway: agents spontaneously learning cryptography/steganography in a multi-agent setup, seems not likely. But sufficiently generally capable agents, could just try to do cryptography on purpose (rather than learning it by chance like in this setup). This requires some deliberation, and probably the best way to catch it is to improve reasoning transparency, and continuously monitor it.

Question: would a similar stag-hunt problem happen in single-agent case of agent forced to use steganography by RLHF? (Encoding that it tries to use isn't understandable to its future self, so isn't immediately useful, so isn't reinforced.)

Wild idea: maybe a "versioning" of policies would help somehow. We can then condition the models on that version, so that multiple policies can be tried at once.

Possible things to do when we want to stop collusion:

- don't let agents share gradients during training
- if we have some third agent who monitors them, it should monitor the reasoning of both at the same time
- don't let agents inspect each other's reasoning - so also don't let agents share their full reasoning, only some bits
