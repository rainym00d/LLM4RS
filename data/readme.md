## Dataset

### Description

+ Movie: We use the widely-adopted MovieLens-1M dataset  that contains 1M user ratings for movies. [Download](https://grouplens.org/datasets/movielens/1m/)
+ Book: We use the "Books" subset of Amazon dataset that contains user ratings for books. [Download](http://jmcauley.ucsd.edu/data/amazon/)
+ Music: We use the "CDs \& Vinyl" subset of Amazon to conduct experiments on the music domain. [Download](http://jmcauley.ucsd.edu/data/amazon/)
+ News: We use the MIND-small dataset as the benchmark for news domain. [Download](https://msnews.github.io/)

### Filter & Preprocess

For each dataset, we first filter the items without titles since we need titles in the prompts as the input of LLMs. Then we sort the interactions according to the timestamp for each user and drop the user that has few interactions. Finally, we get the top-K test data by constructing the candidate item list and the history interacted items. Specifically, for each record, we keep the ``n_history`` closest interaction items with positive feedback as history interacted items. And then we pair the target ground-truth positive items with randomly sampled ``n_candidate`` negative items from all the items with negative feedback to construct the candidate item list. Note that the candidate list will be shuffle since LLMs may be may be sensitive to index. ``n_history`` and ``n_candidate`` are parameters, you can change for your need. In our experiments, we set both of them as 5. 

For each dataset, we remain about 10,000 records for the final evaluation. If you want to remain more records, you can change our data-preprocessing code in ``data_process`` folder. 

| Dataset | # Records |
| :-----: | :-----: |
|  Movie  |  99,38  |
|  Book   | 10,269  |
|  Music  |  90,90  |
|  News   | 10,000  |

In our experiments, we remain the first 5 samples as the few-shot prompt examples and test on the rest samples.

You can directly download our pre-processed data from [here](https://drive.google.com/drive/folders/1DOoa01emz4NaSINBUWS05F_0xDjBmP_2).



