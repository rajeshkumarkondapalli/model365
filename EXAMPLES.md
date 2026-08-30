# Example input and output

The model is a character-level GPT trained on `simple_llm/data/sample.txt`
via next-token prediction. After training it can continue any prompt made
of characters that appear in that corpus.

## Training

```
$ python -m simple_llm.train
step     1  loss 3.5066
step   200  loss 0.4568
step   400  loss 0.2146
step   600  loss 0.1640
step   800  loss 0.1769
step  1000  loss 0.1561
step  1200  loss 0.1405
step  1400  loss 0.1375
step  1600  loss 0.1390
step  1800  loss 0.1711
step  2000  loss 0.1353

generated text:

 what matters. transformers stack attention and feed-forward layers to build deep contextual representations. the model repeats this process to generate one token at a time.
the quick brown fox jumps over the lazy dog. a simple model learns patterns in text by predicting the
```

`python -m simple_llm.train` also writes `simple_llm/checkpoint.pt`, which
`simple_llm/generate.py` loads to continue an arbitrary prompt without
retraining.

## Prompt continuation

```
$ python -m simple_llm.generate "the quick brown" --max-new-tokens 80
input:  'the quick brown'
output: 'the quick brown fox jumps over the lazy dog. a simple model learns patterns in text by predicti'
```

```
$ python -m simple_llm.generate "attention lets each" --max-new-tokens 80
input:  'attention lets each'
output: 'attention lets each token look at other tokens in the sequence to decide what matters. transformers'
```

The model correctly continues each prompt with the text that follows it
in the training corpus — evidence that next-token prediction, attention,
and the transformer stack are working together as intended. Note this is
a tiny model trained on a ~500-character corpus repeated 20 times, so it
has essentially memorized that text rather than generalizing to novel
language; a larger, more diverse corpus would be needed for that.
