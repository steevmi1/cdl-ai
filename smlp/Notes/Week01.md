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

## Satisficing and Optimizing metric

## Train/dev/test distributions

## Size of the dev and test sets

## When to change test/dev sets and metrics

# Comparing to human-level performances

## Why human-level performance?

## Avoidable bias

## Understanding human-level performance

## Surpassing human-level performance

## Improving your model performance
