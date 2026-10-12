# Open questions

Things worth measuring on the model once it trains, rather than arguing about in advance.

**How much does the positional path reach the logits?** With weight tying, `lm_head` is `tok_emb`
transposed, so a logit picks up a term `e_v · p_t` — the dot product of a token embedding with a
position embedding. That is a learned position-conditional prior over the vocabulary, which may be
useful signal or may be leakage. Take a trained checkpoint, compute logits normally, then again
from `h - pos_emb(t)`, and compare the two distributions: KL, or how often the top-1 token agrees.
Near-total agreement means the path carries nothing at the output; 85% means it is doing real work
and removing it would cost you.

Subtracting it in `GPT.forward` is not obviously the fix either way — `ln_f` sits between the
residual stream and `lm_head`, so subtracting before the norm gets partly undone by it, and
subtracting after fights a scale mismatch. The architectures that took this seriously removed
absolute position from the residual stream rather than taking it back out: RoPE rotates `q` and `k`
inside attention and adds nothing to the stream (`reference/roformer-rope.pdf`), and NoPE drops
positional encoding entirely, on the argument that a causal mask already implies order.

**Is random-offset sampling worth its uneven coverage?** `get_batch` draws start offsets at random
with replacement, so a token is visited a Poisson-distributed number of times — at 24 expected
visits, most land between 14 and 34, and none of it is coordinated. A partitioned sampler would fix
that: cut the training split into non-overlapping windows of `block_size`, shuffle them once per
epoch, and every token is seen exactly once per pass. Divisibility then starts to matter, and you
would truncate to a whole number of batches to drop the ragged one.

What the current sampler buys in exchange is a mild free augmentation. A token appears at a
different offset inside the window every time it is drawn, so it is predicted sometimes from three
tokens of context and sometimes from a hundred — different tasks, same data. nanoGPT and Karpathy's
lecture both sample randomly, which is why this repo does.

The experiment: same seed, same config, swap only the sampler, compare validation curves. If
partitioning wins, the augmentation was worth less than the coverage; if it loses, the reverse. One
number either way, and nobody agrees on the answer in advance.
