# Shallow Machine Translation 

Owen Shay

202006664

x2020gft@stfx.ca


## Summary

The highest performing neural machine translation models (NMTs) usually require an extremely high amount of data to train, millions of parallel sentences in source and target languages in order to achieve promising results. 

This makes them weak on languages without vast amounts of available translations and near impossible to deploy into a lightweight application, limiting them to use on high-performance servers, distributed computing services or high performing hardware with many GPUs and CPUs. 

My goal is to create a shallow, low-resource NMT that can be deployed in resource-contrained environments while achieving comparable results to larger NMTs. 

## Dataset Description

I chose to use a reduced version of the [WMT10 French-English Corpus](https://www.statmt.org/wmt10/translation-task.html), a challenge dataset from the Annual Meeting of the Association for Computational Linguistics (ACL). The data is mainly taken from a 2010 release of the Europarl corpus. 

The WMT10 French-English corpus contains around ~1 billion tokens in both french and english with parallel translations. As my goal is to make a lightweight model, the reduced version I am using contains roughly ~1.8 million tokens in english and ~1.9 million tokens in french, which is around ~0.18% of the data. This reduction will ensure that my model can be deployed in lightweight architechture. 

The files contain 137860 parallel sentences of the source language (english) and their target language (french) translations. This structure is provided for ease-of-use when preprocessing the corpora and training/testing the models. 

