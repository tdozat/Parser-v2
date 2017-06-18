# UnstableParser
This is the parsing system that currently (as of 06/18/2017) holds state-of-the-art performance at the CoNLL2017 shared task on Universal Dependency parsing. It's under active development, so new features are likely to break old models, and it contains a bit of experimental stuff that hasn't been published or isn't working yet.

## Training the parser
If you don't care about hyperparameter settings at all, the basic commands to train and run the parser are the following:
```sh
python main.py --save_dir saves/default train
python main.py --save_dir saves/default parse /path/to/treebank/*.conllu
```
Parsed files are by default saved in the save directory with the same name as the original file. You can also specify the output directory with the `--output_dir` flag, and when parsing a single file you can give it a new name with the `--output_file` flag.

Of course, you need to tell the parser where the train and validation data is located, and there are a lot of hyperparameters to play with. You can check them all out in `config/defaults.cfg`, but the most important ones (including the location of the training datasets) have been condensed down into a separate configuration file called `config/template.cfg`. What you probably want to do is make your own copy of `config/template.cfg` (we'll say `config/my_config.cfg`), which you can then freely modify. Any parameters not specified here are loaded in from `config/defaults.cfg`. Once you've tweaked the settings to your liking, you can train a model that uses it with the following command:
```bash
python main.py --save_dir saves/my_model train --config_file config/my_config.cfg
```
The model saves all the configuration settings in the save directory, so you don't need to re-specify this file when running the model:
```bash
python main.py --save_dir saves/my_model parse /path/to/treebank/*.conllu
```

You might want to keep most settings the same but change one or two on the command line without re-editing the configuration file. To do this, you specify `--config_heading setting1=value1 setting2=value2 ...`. For example, to only use the first 500,000 entries of a pretrained embedding matrix (which can speed up loading time for test runs), you would run the following:
```bash
python main.py --save_dir saves/my_model train --config_file config/my_config.cfg \
                                               --pretrained_vocab max_rank=500000
```
Again, the model saves this information, so you don't need to specify it again when parsing:
```bash
python main.py --save_dir saves/my_model parse /path/to/treebank/*.conllu
```

More documentation to come!
