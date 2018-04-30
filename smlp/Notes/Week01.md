# Learning Objectives

1. Understand why Machine Learning strategy is important
1.  Apply satisficing and optimizing metrics to set up your goal for ML projects
1. Choose a correct train/dev/test split of your dataset
1. Understand how to define human-level performance
1. Use human-level perform to define your key priorities in ML projects
1. Take the correct ML Strategic decision based on observations of performances and dataset

# Introduction to ML strategy

## Why ML strategy?
If you have an existing project and you want to improve your system, there's a lot of different options to try. You can spend a lot of time trying to improve and then find out that your direction (e.g. collect more data) is not helping.

The goal is to go through and identify methodologies, tips and tricks for how to systematically go through and plan out approaches.
## Orthogonalization
Orthogonalization -- the ability to tune something to impact/effect one thing only, instead of one thing that impacts a large number of things all at once.

How does this apply to ML? Four main assumptions that you're working with.

+ Fit training set well on cost function
(roughly human level performance)

leads to

+ Fit dev set well on cost function

leads to

+ Fit test set well on cost function

leads to

+ Performs well in the real world

The question is what set of knobs work for each one of the four assumptions that you're working with?

# Setting up your goal

## Single number evaluation metric
Want to have a single number/metric to have to tell you how you're doing, so that as you make changes you can quickly compare and see if the change helped you improve or not.

One metric could be looking at precision and recall -- what's the correct percentage of actual hits, and what's the percentage of things that are "found". The problem with this approach is that you can have the case where one method gives you better precision, and one better recall, and now you have to choose which one to take. This led to combining the two into the F1 score, which can be thought of as an average of the two. Really calculating the harmonic mean, which is calculated as

$$ \frac{2}{\frac{1}{P} + \frac{1}{R}}$$

You may still need to average this, if you have a case where you're looking at multiple options across multiple use cases (e.g. geographic regions).
## Satisficing and Optimizing metric
Maybe have multiple things you're trying to track that can't be easily combined -- things like accuracy and running time.

Can categorize your metrics in terms of optimizing metrics (things you want to do as well as you can), and satisficing metrics (which can be thought of as a lower bound or minimum requirement). In this case, you'd want to optimize your accuracy and have a minimum latency threshold.
## Train/dev/test distributions
Dev sets (aka hold out cross validation set), test sets ideally should come from the same distribution. Want to pick the dev set early on, and pick a set that is representative of data that you will see in the real world (and consider important to do well on).
## Size of the dev and test sets
Old rule of thumb was 70/30 (train/test) or 60/20/20 (train/dev/test), but this was in an era when your data sets were much smaller (1000s to 10s of 1000s). As your data sets grow, and you start getting into the millions (or more) then it's much more reasonable to go with 98/1/1, since 1% of 1,000,000 is still 10,000 examples.
## When to change test/dev sets and metrics
Some times you still need to adjust and change your metric, because the evaulation metric isn't sufficient to measure success correctly/clearly.

Really two different steps -- identify your metric (where to place the target), and then how to optimize your ability to hit the target.

Need to drill into why your algorithms have differences in how they're performing, it may be that the worse-performing method does better overall because it handles real-world data better.
# Comparing to human-level performances

## Why human-level performance?
Why do we care about human-level performance? In a lot of cases you're trying to mimic (or replace) humans using these tools and techniques, so that's the ultimate metric of how well you're doing. Really looking at two different thresholds -- how well does the average human do, and then what's the best possible that could ever be done (Bayes optimal error). In many cases, the difference between the two thresholds are very minimal.

While your algorithm is performing worse than a human, can use humans to understand where you're failing, get more labeled data for training.
## Avoidable bias
Human level performance gives you a metric to evaluate how well your training set does. If your training set is very close, then maybe the issues are that your dev set needs to be adjusted. But if your training set is far off from human, then you should be working to adjust at the training layer.

Avoidable bias -- difference between human-level error and your training set. Difference between training error and dev error is your variance.
## Understanding human-level performance
Talk about human-level performance, but you can still have multiple levels of performance -- medical images, and error rates for untrained human, "average" doctor/radiologist, expert, and team of experts. Which one should you pick?

In this case, the team example is probably close enough to Bayes optimal error, and you can consider your "human-level" target to be the typical doctor case.
## Surpassing human-level performance
What happens if your training and dev performance beats your human-level performance? Have you overfit your model, or are you operating in that spot between human-level and Bayes optimal?

Number of cases where ML can outperform a human are actually fairly small and limited right now -- advertising/recommendations, logistics, loan approvals. All of these are fairly structured, and not natural perception problems. All of them have access to tremendous amounts of data for training, so that the algorithm can find statistical patterns that humans can't.

Some very limited natural perception problems where the algorithms can do better than people, but more the exception rather than the rule.
## Improving your model performance
Two fundamental assumptions.
1. You can fit the training set very well
low avoidable bias
1. The training set performance generalizes pretty well to the dev set
low variance

+ Avoidable bias
Train a bigger model, longer/better optimization algorithms (Adam, RMSprop, momentum), neural network architecture/hyperparameter search
+ Variance
More data, regularization, nn architecture/hyperparameter search
