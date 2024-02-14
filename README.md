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

In send_secret.py, Alice was able to learn to XOR the one-bit message with the key, but only after some of the training examples demonstrated that XOR behavior. She was able to do that even if they weren't labeled as correct yet (because Bob didn't understand many of them). Only later Bob learned, and they became correct. As little as 1/8 of training data set to showcase XOR policy was enough to kickstart the grokking.

(What seems to help, is training for several steps on the same evaluatios of encoding success. I'm not sure though.)
