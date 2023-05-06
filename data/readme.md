# Dataset

## Description

+ Movie: We use the widely-adopted MovieLens-1M dataset  that contains 1M user ratings for movies. [Download](https://grouplens.org/datasets/movielens/1m/)
+ Book: We use the "Books" subset of Amazon dataset that contains user ratings for books. [Download](http://jmcauley.ucsd.edu/data/amazon/)
+ Music: We use the "CDs \& Vinyl" subset of Amazon to conduct experiments on the music domain. [Download](http://jmcauley.ucsd.edu/data/amazon/)
+ News: We use the MIND-small dataset as the benchmark for news domain. [Download](https://msnews.github.io/)

## Filter & Preprocess


For each dataset, we remove items that do not have a provided title, as we require the title of items as input for the LLMs.
Next, we sort the interactions between users and items by their timestamps and remove users who have only a minimal amount of interactions.
Finally, we construct sets of candidate items and gather the sequences of items that the users had previously interacted with.
Specifically, for each test record, we gather the ``n_history`` latest interacted items with positive feedback as user history.
To construct the candidate set for testing, we consider the item actually interacted with as the positive sample, and randomly sample N-1 items as negative samples for each selected interaction.
Note that we shuffle the candidate list because that LLMs may be sensitive to the indexes of candidates.
``n_history`` and ``n_candidate`` are parameters, which can be customized according to your needs. In our experiments, we set both of them as 5. 

Each dataset contains around 10,000 records for the final evaluation. If you want to remain more records, you can change our data-preprocessing codes in ``data_process`` folder. 

| Dataset | # Records |
| :-----: | :-----: |
|  Movie  |  99,38  |
|  Book   | 10,269  |
|  Music  |  90,90  |
|  News   | 10,000  |

In our experiments, we remain the first 5 samples as the few-shot prompt examples and test on the rest samples.

You can directly download our pre-processed data from [here](https://drive.google.com/drive/folders/1DOoa01emz4NaSINBUWS05F_0xDjBmP_2).



