---
date: "2020-06-08"
title: "Unsupervised Text Style Transfer with Deep Learning"
description: Exploration of deep learning architectures used for generative text style transfer
tags:
- Statistics
- Projects
- ML
- PyTorch
---

# Unsupervised Text Style Transfer with Deep Learning

2020-06-08

Natural Language Processing (NLP) is one area of deep learning that continues to make substantial, and often surprising, progress while also adding value to existing businesses. Supervised tasks like neural machine translation produce high quality, real-time results and Gmail's predictive text feature often feels like magic. I have been most interested in recent applications of generative text models, such as using GPT-2 to [play chess](https://slatestarcodex.com/2020/01/06/a-very-unlikely-chess-game/), [write poetry](https://www.gwern.net/GPT-2), or [create custom games](https://aidungeon.io/). In that vein, I embarked on a deep learning project to see whether recent advances in style transfer, applying the style of one text to another while preserving the content, could be employed to increase the sophistication of a piece of writing.

Along with my coauthor Robert Schmidt, we looked into using adversarial autoencoders and transformer models to generate more sophisticated texts. Below I'll outline our approach and conclusions, but our more detailed paper can be found here: [Generative Text Style Transfer for Improved Language Sophistication](http://cs230.stanford.edu/projects_winter_2020/reports/32069807.pdf)

## Autoencoders

After a wide-ranging literature review into unsupervised style transfer, we saw that autoencoders were the most common architecture employed and offered a diverse set of implementations. In its simplest form, an autoencoder consists of two parts: an encoder and a decoder. The encoder takes in a data matrix of a given size and produces its representation in a lower dimensional space. Concretely, its input layer might have 128 units while its output only has 64, forcing it to compress the information contained in the data. The decoder performs the opposite task, taking this lower dimensional, latent representation as input and outputting a reconstruction of the data. In a traditional implementation of an autoencoder, one might use a loss that penalizes differences between the original data and its reconstruction, thus encouraging the model to reproduce a copy of data from the compressed latent representation.

For the purposes of style transfer, there are added levels of complexity built into this basic model. While approaches differ across authors, the underlying idea is to separate the style space from the content space as part of the encoding, then train the decoder to faithfully recreate the content with a different style vector applied. In [Shen et al., 2017](https://arxiv.org/pdf/1705.09655.pdf), decoders are "cross-aligned," meaning they attempt to align the generated sample from one style with the true sample from the other. In [Vineet et al., 2018](https://arxiv.org/pdf/1808.04339.pdf), the authors try to disentangle content and style latent spaces using a classifier to discriminate between styles adversarially. The encoder is trained to create a style-neutral content space by producing representations that leave the classifier unsure; the content is passed to the decoder with a style vector to produce a sentence with altered style.

While there were many such models to choose from, we followed the approach outlined in [Zhao et al, 2018](https://arxiv.org/pdf/1706.04223.pdf), an "adversarially regularized" autoencoder. This model is similar in that it employs a GAN structure to discriminate styles but employs a single loss function across encoder, decoder, and the style discriminator.

## Transformers

While almost all of the papers in unsupervised style transfer published in 2017-2018 made use of autoencoders, we noticed that the newest preprints focused on transformer architectures. Transformers scrap the entire idea of content and style latent spaces - disentangling a sentence into these blocks is prone to error for subtle styles and fails to capture the complexity of the semantics in a limited vector representation. Instead transformers rely on a self-attention mechanism, a method of mapping dependencies among the words in a sentence rather than processing them sequentially. We rely on the Style Transformer proposed by [Dai et al., 2019](https://arxiv.org/pdf/1905.05621.pdf), in which style is seen as a distribution over a dataset. Similar to the adversarial autoencoder, a discriminator is used to attempt to categorize the style of a sentence. Then the content is preserved by passing a generated sentence through the network again, reversing the style transfer and attempting to recreate the original sentence. The sentence is nudged towards the target style by trying to fool the discriminator into assigning the target style as the most likely class.

## Implementation

With some promising candidate models picked out, our work had just begun.  One major challenge was finding datasets that would work well with these models - sophistication is hard to define and we did not simply want to transfer the style of a single author, such as [Shakespeare](https://www.aclweb.org/anthology/W17-4902.pdf). We ended up defining a "naive" dataset composed of relatively high scoring anonymized essays published by the Hewlett Foundation as part of a [Kaggle competition](https://www.kaggle.com/c/asap-aes/) on automated essay scoring. The "sophisticated" dataset was composed of diverse texts from [Project Gutenberg]( https://www.gutenberg.org/) and the [Oxford Text Archive](https://ota.bodleian.ox.ac.uk/repository/xmlui/) that had little dialogue or other features that might break from the author's personal style. Texts were then stripped of common ("stop") words as well as proper nouns using Spacy's [Named Entity Recognition](https://spacy.io/api/annotation#named-entities) (NER) API, though tags were changed to conform to the Stanford NER tags already used to anonymize the Hewlett data. Proper processing of the dataset helped produce more refined results and the authors and modifications were tuned throughout the process.

The embeddings, numerical vector representations of words and sentences, were also a key consideration in how the models operated. We could train embeddings ourselves or use pre-trained embeddings like [GloVe](https://nlp.stanford.edu/projects/glove/). While we tried to use larger embeddings such as [BERT](https://huggingface.co/transformers/model_doc/bert.html), our GPU struggled with the memory required.

Finally, while we could read some of the output to get a sense of our success, we needed a more rigorous way of evaluating the output of these models. We made use of a few common scoring mechanisms, like BLEU and PINC, that calculate the similarity (and dissimilarity) between two sentences. This served as a crude measure of how much the model actually changed the words used in a sentence - too few and the model is mostly useless, too many and the content is likely not preserved. We then turned to the [KenLM](https://kheafield.com/code/kenlm/) language model, which we trained on the sophisticated dataset, allowing us to measure the perplexity of style transferred sentences. Samples with low perplexity were more likely to come from the target language distribution, meaning they better reflected the sophisticated style. Finally we looked at some fluency scores like the [Flesch Kincaid Grade Level](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests) and Flesch Reading Ease indexes that attempt to quantitatively estimate the reading level for a given sentence.

## Conclusions

Most paper implementations of neural style transfer used sentiment as a proxy for style, and it is far easier for a discriminator to classify a sentence as positive or negative than pick up on sentence structure or formal arguments. While that might seem trivially true, the differences between naive and sophisticated texts were large and easily distinguishable for a human. Both the transformer and autoencoders are not yet prepared to distinguish more nuanced differences in language.

On the other hand, it was clear that the transformer was a real improvement over the older autoencoder models. Its sentences were more coherent and more clearly reflected some of the sophisticated style we were trying to capture. Given the novelty of many transformer models, it seems reasonable to expect continued progress towards capturing subtleties in language without simply scaling up the compute needed.

All of the code and processing used for this project can be found on [Github](https://github.com/spencerbraun/sophisticated_style_transfer).
