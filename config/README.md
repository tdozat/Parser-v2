# UnstableParser/config

## Overview
This is the de facto directory for configuration files. There are a handful of things you should know before playing around with them:

* If you provide an underspecified config file to the model, it will fill inherit everything not specified from `defaults.cfg`.
* Every python class should have a (possibly empty) section in a config file, either `defaults.cfg` or a user-provided one. Uppercase letters with lowercase letters preceding or following them should be separated from preceding letters with a space.
      * class BaseXTagger -> [Base X Tagger]
      * class RNNEmbed -> [RNN Embed]
* If class B inherits from class A, then any options for class B not present in section [B] will be inherited from section [A].
* Options must be manually assigned a type (int, float, boolean, list) in `parser/configurable.py`, which basically everything in the codebase inherits from.
* list elements should be separated by colons
      * words:tags:xtags
      * /scr/tdozat/PTB/UD/train.conllu:/u/nlp/data/CoNLL17/UD_English/en-ud-train.conllu
* (lists of) files support globbing
      * /scr/tdozat/PTB/UD/*.conllu
      * /scr/tdozat/PTB/UD/\*.conllu:/u/nlp/data/CoNLL17/UD_English/\*.conllu
* 'None' is always `None` 

## Config Options
### Base
#### DEFAULT
Default options accessible to all sections in the configuration file

* `save_dir`: The directory where the model, any auxiliary files, and default output should be located
* *everything else*: Convenience variables that can be used in other sections, such as language codes

#### Configurable
Options accessible to all classes

* `train_files`: List of training files
* `parse_files`: List of validation files
* `verbose`: Boolean indicating whether to print everything or just some things
* `name`: String identifier used to name auxiliary files, tensorflow variable scopes, and distinct instances of the same class

### Vocabulary
#### Base Vocab
Default options inherited by all derivative vocabulary classes. Basis for Pretrained Vocab and Token Vocab.

* `cased`: Boolean indicating whether to use cased or uncased words
* `embed_size`: Dimensionality of the embedding vectors

#### Pretrained Vocab
Vocabulary designed for reading and accessing pretrained embedding matrices

* `special_tokens`: List of tokens reserved for special purposes, such as padding or replacing OOV words
* `skip_header`: Some pretrained embeddings use the first line of the embedding file for recording metadata, which we want the model to ignore; setting this flag to `True` discards this metadata
* `filename`: Location of the pretrained embedding matrix file, as uncompressed text or text compressed with `xz` (support for gzip is coming!)
* `max_rank`: How many embeddings to read in. For large embedding files, setting this to 1,000,000 or less accelerates the startup time

#### Token Vocab, Word Vocab, Lemma Vocab, Tag Vocab, X Tag Vocab, Rel Vocab
Vocabulary designed for holistic, frequent word embeddings

* `embed_keep_prob`: Probability of dropping the embedding during training
* `min_occur_count`: How many times a word must occur to be considered 'frequent'
* `filename`: Where to save the token count frequencies

#### Index Vocab, Dep Vocab, Head Vocab
Vocabularies designed to store indices, such as the token's index in the sentence or the index of its head

#### Subtoken Vocab, Char Vocab, ...(experimental)
Extension of Token Vocab for subtoken sequences (e.g. sequences of characters or n-grams)

* `n_buckets`: How many buckets to sort the sequences into (lower values decrease runtime, higher values decrease memory demands)
* `embed_model`: What kind of network to use for computing embeddings from sequences; must be the name of a BaseEmbed class
* `embed_keep_prob`: Probability with which to drop subtokens

#### Multivocab
Primarily used for aggregating together pretrained word embeddings, frequent-token word embeddings, and character-based word embeddings

* `embed_keep_prob`: probability with which to drop (aggregated) embeddings

### Neural models
#### NN
Class with essential neural functions, such as MLPs and attention

* `recur_cell`: What kind of recurrent cell to use; must be the name of a BaseCell
* `n_layers`: How many layers to use
* `mlp_func`: Name of the nonlinearity to use; must exist in `parser/neural/functions.py`

(Under construction)
