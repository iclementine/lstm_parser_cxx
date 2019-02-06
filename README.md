# how to generate a oracles 

```bash
export PYTHONPATH=/home/clementine/projects/
python transduce_corpus.py -t 'arc-standard-swap' -p xpos -f normal train.conll > train.txt
python transduce_corpus.py -t 'arc-standard-swap' -p xpos -f normal dev.conll > dev.txt
python transduce_corpus.py -t 'arc-standard-swap' -p xpos -f normal test.conll > test.txt
```

# how to train a model
