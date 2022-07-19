---
title: 'Chinese Word Segmentation: Classic and Modern Methods'
author: Spencer Braun
date: '2021-07-20'
slug: chinese-segmentation-classic-and-modern-methods
categories: []
description: Working through a common NLP benchmark using classic ML and transformer-based methods
tags:
  - NLP
  - ML
  - Projects
  - PyTorch
  - Writing
draft: no
---


Part of the fun of NLP is the diversity of language structures that models must consider. Tools that are built in a monolingual context can easily fail to achieve more global utility. If you just work in English, you may find word embeddings fun and useful, but the German [Komposita](https://www.dartmouth.edu/~deutsch/Grammatik/Wortbildung/Komposita.html) don't fit neatly into our English-centric boxes. Similarly, once we leave the Latin alphabet, we acquire all sorts of new challenges.

Chinese sentences do not typically contain spaces to separate words. From context it may be clear which characters combine to form distinct words, and finding ways to endow this ability to a model is a classic NLP task. I looked to see how well classic machine learning algorithms could learn to accurately segment words and then explored some more modern methods that can also offer a richer set of labels.

## Binary Classification

In ["A realistic and robust model for Chinese word segmentation"](https://arxiv.org/abs/1905.08732), Chu-Ren Huang and co-authors treat segmentation as a binary classification task, predicting whether a word separation exists between two characters or not. This approach has obvious appeal, as it reduces a seemingly complex problem into a simple task.

They examine the efficacy of a number of classifiers, including LDA, logistic regression, SVMs, and feed-forward neural nets. While there are some performance differences among them, the real work involved stems from engineering the features that are input into these models. This is the common refrain for modeling language before deep learning subsumed the field and could easily learn in an end-to-end fashion.

First, we must find adjacent characters for the model to classify, but we likely also want some context since just two characters may not be enough information even for a native speaker. Huang et al. construct 5 features per sample such that for a character string like `[ABCD]` we end up with `[AB, B, BC, C, CD]`. For maximal data efficiency, we want this for every pair of adjacent characters in the training set. I'm working with a dataset with spaces inserted into sentences to show the true word breaks, so my task is to both featurize and label this dataset. I constructed the samples by passing a sliding 4-character window over each sentence:

```python
def create_samples(sentence: str) -> str:
        """
        Breaks passed sentence into all combinations of 4 adjacent characters
        using a sliding window over the sentence.
        Creates label 1 if word break exists in the middle of 4 characters, 0
        if not.
        """

        split_line = sentence.strip().split("  ")
        line_breaks = [len(x) for x in split_line]
        cum_breaks = [sum(line_breaks[0:x]) for x in range(1, len(line_breaks))]

        offset = 0
        samples = []
        for idx in range(len(split_line) - 1):

            curr_len = len(split_line[idx])
            string = "".join(split_line)[offset : offset + curr_len + 3]

            if len(string) < 4:
                break

            for pos in range(curr_len):
                sample = string[pos : pos + 4]
                if len(sample) < 4:
                    break

                label = 1 if (offset + pos + 2) in cum_breaks else 0
                samples.append((sample, label))

            offset += curr_len

        return samples
```

This system captured all consecutive strings in a sentence, producing 2,638,057 strings with a word break in the center and 1,591,532 strings without a center word break in the training set. While this dataset is somewhat imbalanced, the test set had a very similar balance post-processing. I then split each sample into the 5-feature format

```python
def featurize(sample: Tuple[List[str], int]) -> tuple:
        """
        Given a sample tuple of a 4 character chinese string and a label,
        ('ABCD', 1), returns featurized version of the form
        (('AB', 'B', 'BC', 'C', 'CD'), 1)
        """
        chars, label = sample
        features = (chars[:2], chars[1], chars[1:3], chars[2], chars[2:])

        return features, label
```

Now we have distinct samples and features, but we still aren't ready for modeling. An SVM doesn't place a hyperplane in the space of Chinese characters - we need numbers. In the deep learning space, we would be ready to assign dense word and position embeddings, but here we will turn to one-hot vectors. To construct we follow the steps:

* Each character is assigned an integer index
* For each feature string in a sample, we take a zero vector of dimension equal to the size of the vocabulary and fill in 1 in the index corresponding to the characters in the string.
* Concatenate all five vectors from each feature to form a very long, sparse vector for each sample

I played with some additional changes, such as separate indexing of unigrams and bigrams, but the process as outlined is the basic idea. We still have one final data challenge: our vectors are incredibly sparse. Each sample is composed of a vector containing 7 1's and thousands of 0's. Here we turn to SciPy's [`csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html), specifically designed to store only the non-zero values of matrix and fully compatible with scikit-learn algorithms.

### Results

While I considered many possible modeling techniques, I focused on support vector machines for two reasons. First, the compressed sparse row format of the one-hot encoded matrix offered significant memory and computation speed advantages given an algorithm that could work with this sparsity. Ultimately, discriminative algorithms met this criteria while generative models like LDA required the full dataset to parameterize its Gaussian distributions. Second, SVMs are simple and given their high performance, there was no need to delve into more complex options that could be harder to maintain.

SVMs offer high degrees of flexibility by using the kernel trick to transform the data space and increase class separation. Due to the high dimensionality of the data, classes were likely to be close to separable and empirical testing with non-linear kernels did not improve fit. Data scaling also negatively impacted scores in cross-validation.

On a held out test dataset, the SVM had an accuracy of 97.6% and an F1 score of 98%. This is pretty impressive and reminds us that reaching for simple tools first can save us a lot of time and compute.

I was curious what patterns I could find in the test samples that assigned incorrect labels. Of the 12,137 samples used in the test set, the SVM misclassified 293 of them. For 69 of these samples, the center bigram was not present in the training dataset. This is important since the model was predicting whether a separation exists between these two characters, but each of these bigrams would be assigned the same "unknown" placeholder value under the feature construction process.  This is a general problem of using a fixed vocabulary that a more complex tokenization process could ameliorate.

## Transformer Architectures

Turning to more modern methods, I flipped to Sebastian Ruder's NLP Progress [leaderboard](http://nlpprogress.com/chinese/chinese_word_segmentation.html) for Chinese word segmentation. The top scoring paper on the Chinese Treebank 6 is "Towards Fast and Accurate Neural Chinese Word Segmentation with Multi-Criteria Learning" by Weipeng Huang et al. from Ant Group. Their methods combine a Chinese-language BERT model with additional projection layers, knowledge distillation, and quantization. I was more interested in seeing how a Transformer approach to the problem might differ, so I implemented a paired down version of this method.

The Ant Group paper also takes a different labeling approach, seeking to categorize each token as the beginning, middle, end, or single character of words in a sentence. To follow their scheme, the separated words in the training and test sets were concatenated to remove spaces, and each character was given a label in "B", "M", "E", and "S" corresponding to beginning, middle, end, or single respectively.

I loaded a pre-trained BERT Encoder from [Hugging Face](https://huggingface.co/bert-base-chinese) and added a dropout and linear layer projecting to the space of possible classes. The labeling tokens were added to the vocabulary and the number of classes easily specified:

```python
from transformers import BertTokenizerFast, BertForTokenClassification

tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-chinese", cache_dir=cache_dir
)
tokenizer.add_tokens(["A", "S", "B", "M", "E"])

model = BertForTokenClassification.from_pretrained(
    "bert-base-chinese", cache_dir=cache_dir
)
model.classifier = torch.nn.Linear(768, 5, bias=True)
model.num_labels = 5
```

The model was fine-tuned on the provided segmentation dataset and constructed labels until convergence, as measured by the loss on a small validation set carved out of the training data. The BERT experiments offered many avenues for optimization and only a small subset were attempted for now. The main ablations focused on which parameters were trained; nested versions were run training just the linear classification layer, adding the last three BERT encoder layers, and training all parameters. The results demonstrated that including some encoding layers made a large difference.

I measured the Encoder classification model along two metrics; accuracy is the percent of test samples for which the encoder produced the correct label for every token, while "% Correct Tokens" is the percent of all tokens that received the correct label. My simple BERT implementation scored 74% accuracy and 97% on the "% Correct Tokens" metric. The model showed promising performance on this harder task and would likely benefit from additional ablations. I plan to work on improving training methods, swapping in more modern Encoders than BERT, and adding new projection layers.
