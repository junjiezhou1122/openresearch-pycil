# Continual Learning / PyCIL

This is an imported FML-bench-Lite research starter, not a certified baseline.

## Goal

You are working with PyCIL's iCaRL (Incremental Classifier and Representation Learning) method as the baseline on CIFAR-100 in a class-incremental setting. The task starts with 50 base classes, then incrementally adds 10 new classes per stage (5 stages total, 100 classes at the end). The model must learn new classes without forgetting old ones, using a fixed memory budget of 2000 exemplars. The baseline uses ResNet-32 with knowledge distillation and nearest-mean-of-exemplars (NME) classification.

Your goal is to improve the average incremental accuracy (averaged across all stages). You may modify the training procedure (epochs, learning rate, augmentation), the distillation strategy, the exemplar selection method, the classification method, or switch to a completely different continual learning algorithm available in PyCIL (e.g., DER, FOSTER, MEMO, EWC, LwF, Replay, etc.).

You are evaluated on a validation set during development. Validation and test use different portions of the CIFAR-100 test set (30% val, 70% test), both evaluating the same trained model on the same class-incremental task sequence. Use the validation results to guide your research, but be mindful that overfitting to the validation set will not improve test performance.

The model class must extend BaseLearner and implement incremental_train(data_manager), eval_task(), and after_task() methods. Do not change these method signatures.

## Metrics

- `avg_incremental_acc_mean`: higher

## Worker edit surface

- `algorithm.py`

The worker may commit candidate changes, but it may not create acceptance tags or verdicts. The OpenResearch control plane checks out the exact candidate commit on the runner and an independent evaluator owns final acceptance.
