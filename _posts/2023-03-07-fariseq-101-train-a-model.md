---
layout: distill
title:  Fairseq 101 - train a model
date:   2023-03-07 12:30:00
description: Train your first Fairseq model - tutorial for NLP@WUT class
tags: fairseq nmt tutorial

authors:
  - name: Mateusz Klimaszewski
    affiliations:
      name: WUT

toc:
  - name: Installation
    subsections:
    - name: Fairseq
    - name: SentencePiece
    - name: Wandb
  - name: From data to model
    subsections:
    - name: Parallel corpora
    - name: Training a SentencePiece model
    - name: Data pre-processing
    - name: Training a NMT model
---
Fairseq is my go-to library when it comes to Neural Machine Translation.
The codebase is quite nicely written, and it is easy to modify the architectures.
However, the documentation is suboptimal and, most of the time, does not follow the rapid changes in the new releases.
Code is the best (and the only) documentation in this case.
I decided to write a step-by-step pipeline to ease the first steps with the library (following students' comments about the lack of end-to-end tutorials).
This tutorial aims to train an NMT model from scratch, explaining requirements in terms of libraries, how to get data, and introducing the reader to basic Fairseq commands.
On this fundamental level, the tutorial should be correct even with future releases of Fairseq.

The audience of this tutorial is students taking part in an NLP class @ WUT, whose project is related to Neural Machine Translation.
However, the content is generic and might serve all new Fairseq users.

The interactive (but less detailed) version is available as a [Google Colab notebook](https://colab.research.google.com/drive/1xVRbJiwNRavTnCvLswEA9Nu6jkoSAHax?usp=sharing).
I suggest running it in parallel with reading this tutorial.
The [Machine Translation](https://web.stanford.edu/~jurafsky/slp3/13.pdf) chapter from [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) book by
Dan Jurafsky and James H. Martin should be sufficient as a prerequisite.

# Installation
First, we must install two main elements (optionally 3).
1. Fairseq :wink:
2. SentencePiece - token segmentation
3. Wandb - metrics visualisation (optional, but highly recommended)

## Fairseq
There are two main approaches
* using Fairseq as a black-box tool (`0.12.2` is a specific version for the reproducibility of this tutorial):
```bash
pip install fairseq==0.12.2
```
* writing own models/making changes in the core library:
```bash
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
git checkout v0.12.2
pip install -e .
```

Both require a pre-installed correct [PyTorch](https://pytorch.org/get-started/locally/) version
(Colab has it done for you by default).

## SentencePiece
Fairseq has multiple internal/external possibilities for token segmentation (see `@register_bpe` annotation in the source code).
To list a few:
* subword_nmt
* fastbpe
* sentencepiece

My go-to approach is the [SentencePiece](https://github.com/google/sentencepiece) library.
I prefer to use it as a cmd tool; however, it has Python bindings that can be installed via `pip`.

```bash
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
git checkout v0.1.97
mkdir build
cd build
cmake ..
make -j $(nproc)
```

The executables are stored in the `build/src` directory.
I usually keep the path to this directory in an external environment variable, i.e. `SPM`.

```bash
export SPM=$PWD/sentencpiece/build/src
```

We can test our installation by running the following command:
```bash
$SPM/spm_train --help
```

## Wandb
We need to do a simple [setup](https://docs.wandb.ai/quickstart#set-up-wb) to allow Fairseq to push the metrics into [Wandb](https://wandb.ai/).
For now, this is all; the rest will be done during the execution of the training.

# From data to model
Once again, we need a few steps to reach our goal of training your first Fairseq model. The required steps are:
1. Getting parallel corpora
2. Training a SentencePiece model
3. Pre-processing the dataset
4. Training a model (at last!)

## Parallel corpora
The most common parallel corpora repository is [OPUS](https://opus.nlpl.eu/).
Additionally, more data and neat validation/test datasets can be found on the WMT competitions website (e.g. [WMT22](https://www.statmt.org/wmt22/translation-task.html), [WMT21](https://www.statmt.org/wmt21/translation-task.html)).

It is important to note that some parallel corpora might require additional filtering (e.g. ratio-based, length-based).

```bash
wget -O de-en.txt.zip https://opus.nlpl.eu/download.php?f=News-Commentary/v16/moses/de-en.txt.zip
unzip de-en.txt.zip
```
The archive contains at least two files, one with German sentences and one with English ones.
The format is straightforward - sentence per line (note: automatic sentence segmentation might be mistaken; therefore, the mentioned filtering).

First three lines of the `News-Commentary.de-en.en` file.
<blockquote>
$10,000 Gold?<br>
SAN FRANCISCO – It has never been easy to have a rational conversation about the value of gold.<br>
Lately, with gold prices up more than 300% over the last decade, it is harder than ever.
</blockquote>
The`News-Commentary.de-en.de` has corresponding sentences in German.

## Training a SentencePiece model
We will use the `spm_train` command to train a SentencePiece model.

```bash
$SPM/spm_train --input="News-Commentary.de-en.en,News-Commentary.de-en.de" \
    --vocab_size=16000 \
    --character_coverage=1 \
    --num_threads=8 \
    --max_sentence_length=256 \
    --model_prefix="spm" \
    --model_type=unigram \
    --bos_id=0 --pad_id=1 --eos_id=2 --unk_id=3
```

Few words of explanation for the options used in the command.
The vocabulary size (`--vocab_size`) defines the size of the output vocabulary.
Character coverage (`--character_coverage`) specifies the percentage (in the range of 0-1) of the characters that are present in the final vocabulary.
For alphabets with many symbols, one might consider lowering the value.
Maximum sentence length (`--max_sentence_length`) skips sentences longer than provided value, while model type (`--model_type`) specifies the training algorithm.
Token indices (`--bos_id`, `--pad_id`, `--eos_id`, `--unk_id`) are set to match Fairseq settings.

See other options with documentation by running:
```bash
$SPM/spm_train --help
```

**Important** We need to pre-process the outcome dictionary from the SentencePiece to match the required Fairseq format.
```bash
cut -f1 spm.vocab | tail -n +5 | sed "s/$/ 100/g" > dict.txt
```
This command removes unnecessary lines and replaces values with a constant.

*Before* `spm.vocab` first lines:
```
<s>	0
<pad>	0
</s>	0
<unk>	0
,	-3.12151
.	-3.35171
s	-3.7291
```

*After* `dict.txt` first lines:
```
, 100
. 100
s 100
```
The `dict.txt` file will be the one that we later pass to Fairseq commands as an argument.

## Data pre-processing

Here, we have two tasks to do:

* Using the SentencePiece model to pre-process our data.
* (Partially optional) Binarise the data to the Fairseq format.

### SentencePiece encoding
The first step should be done for all the datasets, including validation and test ones.
The operation will segment the words into subwords, adding the special symbol `▁` to the first subword of a word and a space between subwords.

The command is `spm_encode` with arguments as follows:
```bash
$SPM/spm_encode --model="spm.model" --output_format=piece < "News-Commentary.de-en.en" > train.en-de.spm.en
$SPM/spm_encode --model="spm.model" --output_format=piece < "News-Commentary.de-en.de" > train.en-de.spm.de
```

The results compared to the input:
<blockquote>
$10,000 Gold?<br>
▁$1 0,000 ▁Gold ?
</blockquote>

The algorithm split the `$10,000` into two subwords: `$1` and `0,000`. The first subword is marked with the special symbol `▁`.
The token `Gold` was not split as the SentencePiece algorithm had enough depth (`16000`) to include the word in the dictionary.
However, it was separated from the question mark sign.

All four subwords are separated by space.

### Binarisation
Here, we map the data into the Fairseq format.

First, the default approach, with binarisation.
Note that we provide **external** (unused during SentencePiece training) validation dataset encoded with the trained model.
We also provide created earlier `dict.txt`, assume the `en->de` translation direction and define the BPE algorithm (`sentencepiece`).
The joined dictionary option (`--joined-dictionary`) means we trained just one SentencePiece model for both languages.
It is possible to provide separate models for source and target languages.

```bash
fairseq-preprocess \
    --trainpref "train.en-de.spm" \
    --validpref "valid.en-de.spm" \
    --destdir "bin" \
    --joined-dictionary \
    --srcdict "dict.txt"\
    --source-lang "en" \
    --target-lang "de" \
    --bpe sentencepiece \
    --workers 8
```

Checking the output files, we can see `*.bin` and `*.idx` files in `bin` directory in unreadable, binary format.
However, having the data in readable format might be nice, primarily for debugging.
To achieve that, use the `--dataset-impl "raw"`.
By default, this flag has the value `mmap`. 

## Training a NMT model
The longest part - make sure to have GPU enabled. The provided hyperparameters may be fine, but only may. You might want to enable half precision (`--fp16`) or define/use a smaller model to speed up the training.

In the case of Colab timing out, you can change the `--keep-interval-updates` and `--no-epoch-checkpoints` flags to save intermediate checkpoints and resume the training from the last checkpoint.

```bash
fairseq-train \
        "bin" \
        --fp16 \
        --arch transformer_wmt_en_de \
        --share-all-embeddings \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.0 \
        --lr 5e-4 \
        --lr-scheduler inverse_sqrt \
        --warmup-updates 4000 \
        --warmup-init-lr 1e-07 \
        --dropout 0.1 \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --save-dir "model_output" \
        --log-format json \
        --log-interval 100 \
        --max-tokens 8000 \
        --max-epoch 100 \
        --patience 5 \
        --seed 3921 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok space \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-print-samples \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric
```

After a dozen minutes (or a few hours - depending on the data/GPU/model), we have our model in the `model_output` directory :tada:

### Wandb notes:
Set `--wandb-project` to specify the Wandb project.

You can customise Wandb tags and the run name by setting env variables (`WANDB_TAGS` and `WANDB_NAME`)
