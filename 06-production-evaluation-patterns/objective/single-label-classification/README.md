# Single-Label Classification

This is the simplest objective evaluation pattern. Each input maps to exactly one correct label. Examples: sentiment analysis (positive/negative), intent classification (one ticket → one department), routing an email to one team, or topic categorization (one article → one topic).

## Three flavors, one family

Classification problems come in three shapes that are often confused. They differ only in the label space:

- **Binary classification**: two classes (`positive`/`negative`, `spam`/`not spam`, `refund`/`not refund`). Special case of single-label.
- **Single-label multi-class classification**: more than two classes, but each item still gets exactly one label. Intent classification with 11 possible intents falls here.
- **Multi-label classification**: each item can have multiple labels at once. A single customer feedback message might be both `billing` and `urgent` and `refund`. Different metrics, different evaluation pattern; see [`../multi-label-classification/`](../multi-label-classification/README.md).

This page covers the first two. The metrics and intuitions are the same; only the number of classes changes.

## Why this is different from traditional ML

If you have done classification before with scikit-learn or a deep learning framework, you are used to a specific workflow: collect labeled data, split it into training, validation, and test sets, train a model on the training set, tune hyperparameters on validation, and report final metrics on the held-out test set. The whole point of the split is to prevent the model from memorizing the data it will be evaluated on.

GenAI classification breaks this workflow because **you are not training the model**. You are using a large language model that has already been trained on a vast corpus, and you are asking it to classify new inputs by way of a prompt. There is no fitting step, no gradient updates, no risk that your evaluation data leaks into training. This means:

- **You don't need a train/validation/test split.** Every labeled example you have can go directly into your evaluation set. You are not "wasting" data on training.
- **The evaluation set is the only lever you have for measuring quality.** In traditional ML, a held-out test set is one signal among several (training loss, validation curves, learning rate behavior). In GenAI classification, the evaluation set *is* the signal. If your evaluation set is wrong, you are flying blind.
- **Iteration looks different.** You don't tune hyperparameters; you change the prompt, swap the model, or adjust few-shot examples. Each of these is a "version" that you re-evaluate against the same held-fixed evaluation set, exactly the way you would compare two models on the same test set in classical ML.

The practical implication: every hour you would have spent splitting data and tuning hyperparameters in classical ML, you should now spend on building, curating, and protecting your evaluation set. The rest of this README is built around that assumption.

## Precision and recall, in plain English

Most developers meet "precision" and "recall" and immediately confuse the two. The fastest way to remember them is with a concrete scenario.

Imagine you build a customer support agent that flags incoming tickets as `REFUND` requests. Out of 1,000 tickets in a day, your agent flagged 100 as `REFUND`. Of those 100 flagged tickets, 80 were actually refund requests; the other 20 were misclassified (the customer was asking about something else). Meanwhile, looking at the full 1,000 tickets, 120 of them were genuinely refund requests; your agent found 80 of them and missed the other 40.

That single example contains everything you need.

**Precision** answers the question: *"When my agent says `REFUND`, how often is it right?"* In our example, the agent flagged 100 tickets as refund, and 80 of those were correct. Precision = 80 / 100 = 80%. High precision means the agent rarely cries wolf. Low precision means a lot of false alarms.

**Recall** answers the question: *"Of all the actual `REFUND` tickets, how many did my agent catch?"* In our example, there were 120 real refund tickets in total, and the agent caught 80. Recall = 80 / 120 = 67%. High recall means the agent rarely misses real cases. Low recall means quiet failures: real refund requests slip through unflagged.

Now the pivot every developer needs to internalize: **precision and recall trade against each other, and the right balance depends on what your business pays for failure.**

- **A spam filter** wants high precision. Falsely flagging an important email as spam is much more painful than letting one spam message through, so you tune the agent to only flag when very confident, even if it misses some borderline spam.
- **A fraud detector** wants high recall. Missing a real fraud case costs the business money, so you tune the agent to flag aggressively, even if humans then have to dismiss some false alarms.
- **A medical screening test** also wants high recall, for the same reason: missing a sick patient is far worse than a false alarm that gets resolved with a follow-up test.

The single number that combines precision and recall is **F1 score**: the harmonic mean of the two. F1 of 1.0 means both precision and recall are perfect; F1 of 0 means at least one of them is zero. F1 punishes imbalance: an agent with 100% precision and 10% recall does not get F1 = 55%; it gets F1 = 18%, because the harmonic mean weighs the smaller number more heavily. That is exactly the property you want when neither precision nor recall alone tells the full story.

For multi-class problems (like the 11-intent ticket classifier later in this README), you compute precision, recall, and F1 *per class*, then aggregate. The aggregation choice is what `macro-F1`, `micro-F1`, and `weighted F1` are about. We will get to those in the metrics table; for now, internalize the per-class definitions and the trade-off.

## Why accuracy lies on imbalanced data

Accuracy is the metric every developer reaches for first because it sounds intuitive: "what fraction of my predictions were right?" On balanced data this works fine. On real production data, it almost always misleads you. Here is why, with numbers.

Suppose you build an agent to flag fraudulent transactions. In a normal week, your business processes 10,000 transactions, of which 100 are actually fraud (1% fraud rate, which is already on the high side for retail). Now imagine the world's laziest classifier: it always predicts "not fraud", regardless of input. What is its accuracy?

It correctly classifies all 9,900 legitimate transactions and incorrectly classifies all 100 fraud cases. Accuracy = 9,900 / 10,000 = **99.0%**.

A 99% accurate fraud detector that catches zero fraud is a disaster, but accuracy alone makes it look like a triumph. This is not a contrived edge case. Any time one class dominates the data, accuracy will be high simply by predicting the majority class, regardless of whether the agent has learned anything useful.

Now look at the same scenario through precision and recall:

- The lazy agent flagged zero transactions as fraud, so precision is undefined (you cannot divide by zero), but conventionally we report it as 0 or N/A. There is nothing to be precise *about*.
- The lazy agent caught 0 of 100 fraud cases, so **recall = 0%**. Recall makes the failure obvious immediately.

The same trap appears in subtler forms. A 9-class intent classifier where one class accounts for 40% of traffic can show 75% accuracy while completely failing on the other 8 classes; that 40% class alone explains the headline number. The fix is to never rely on a single aggregate. Always read accuracy alongside per-class precision, per-class recall, and per-class F1, and prefer **macro-F1** when class importance is roughly equal.

Three signals tell you accuracy is misleading on your specific dataset:

1. **Class distribution is uneven.** If your most frequent class is more than twice as common as your least frequent class, accuracy is already at risk.
2. **Per-class F1 spreads widely.** If your best class has F1 = 0.95 and your worst has F1 = 0.4, the aggregate hides the failure.
3. **The classes you care about most are the rare ones.** Fraud, escalations, complaints, churn signals: the rare class is usually the one that actually matters. Accuracy systematically underweights exactly the cases your business depends on.

When any of those signals are true, treat accuracy as a sanity check, not as the headline.

## Choosing your metrics

Now that you have precision, recall, F1, and the imbalance problem in your head, the metrics table is a quick reference rather than a primer. Each row tells you what the metric measures, when it is the right pick, and what it tends to hide.

| Metric | What it measures | When to use | What it hides |
|---|---|---|---|
| **Accuracy** | Fraction of predictions that are correct | Classes are roughly balanced (≤ 2:1 ratio) and all classes equally important | Failures on rare classes when one class dominates |
| **Per-class precision** | For class X, how many of the agent's `X` predictions were correct | You care specifically about false positives for that class | Recall on that class |
| **Per-class recall** | For class X, how many of the actual X items the agent caught | You care specifically about misses for that class | Precision on that class |
| **Per-class F1** | Harmonic mean of per-class precision and recall | A single per-class number that punishes a precision/recall imbalance | Whether the failure is precision-side or recall-side; always read F1 alongside the components |
| **Macro-F1** | Average of per-class F1, equal weight per class | Imbalanced data, all classes matter equally (default headline for most production classifiers) | Overall correctness when one class dominates traffic |
| **Micro-F1** | F1 on pooled (item, label) decisions across classes | Imbalanced data, but you care more about getting most predictions right than treating every class fairly | Per-class behavior |
| **Weighted F1** | Average of per-class F1 weighted by class frequency | You want one number that reflects production class distribution | Whether minority classes are working at all |

**Rule of thumb:** For any imbalanced dataset, report macro-F1 *and* per-class F1. The macro-F1 catches whether you are doing equally well across classes; the per-class F1 catches *which* class is the problem.

## A worked example: support ticket intent classification

Consider a customer support agent that classifies incoming tickets into one of 11 intent categories: `ACCOUNT`, `CANCEL`, `CONTACT`, `DELIVERY`, `FEEDBACK`, `INVOICE`, `ORDER`, `PAYMENT`, `REFUND`, `SHIPPING`, and `SUBSCRIPTION`. We evaluate on 220 samples drawn from the public [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset), with 20 samples per class (a deliberately balanced evaluation set).

Note that production traffic for this kind of agent is rarely balanced. Real ticket distributions are skewed, with a few intents dominating and a long tail of rare categories. We balance the *evaluation* set on purpose so that every class gets equal scrutiny, then read macro-F1 (not accuracy) as the headline number because macro-F1 reports our average performance per class regardless of how often the class shows up in production.

A zero-shot Claude Sonnet classifier produces these aggregate numbers:

- **Accuracy:** 82.3%
- **Macro-F1:** 0.81

If you stopped here, you would ship. But the per-class F1 tells a different story:

| Class | F1 | Notes |
|---|---|---|
| INVOICE | 1.00 | Perfect |
| SUBSCRIPTION | 0.98 | Strong |
| PAYMENT | 0.93 | Strong |
| REFUND | 0.91 | Strong |
| CONTACT | 0.91 | Strong |
| ACCOUNT | 0.82 | Good |
| FEEDBACK | 0.79 | Acceptable |
| CANCEL | 0.74 | Borderline |
| ORDER | 0.72 | Borderline |
| SHIPPING | 0.60 | Weak |
| **DELIVERY** | **0.56** | **Failing** |

![Per-class F1 bar chart sorted descending, with INVOICE at 1.00 and DELIVERY at 0.56](./figures/fig2_per_class_metrics.png)

The per-class view surfaces what the aggregate metrics hide: `DELIVERY` and `SHIPPING` are getting confused with each other, and `ORDER` is being confused with `CANCEL`. Looking at the confusion matrix confirms this: 8 out of 20 `DELIVERY` tickets were misclassified as `SHIPPING`, and 6 out of 20 `ORDER` tickets were misclassified as `CANCEL`.

![Confusion matrix for the 11-class Bitext intent classifier, with the DELIVERY-to-SHIPPING and ORDER-to-CANCEL off-diagonal cells highlighted](./figures/fig1_confusion_matrix.png)

This is the production-relevant finding. Your agent is not 82% accurate uniformly; it has specific class-pair confusions that need attention. Possible fixes: clarify the class boundary in the prompt, add few-shot examples that distinguish the two, or merge the classes if the business doesn't actually need to distinguish them.

## When the classifier is an LLM

Classical ML classifiers (logistic regression, gradient boosting, fine-tuned transformers) produce a calibrated probability per class, and they behave the same way every time you run them on the same input. LLM classifiers don't, and that changes how you evaluate them. Four things to watch:

- **Output variance.** Same prompt, same input, different runs can give different labels, especially at the class boundaries. Set `temperature=0` for evaluation runs, but expect residual variance from sampling. Run each evaluation sample 3 times and report the majority label plus a stability rate.
- **Prompt sensitivity.** Changing the class definition by one word can swing per-class F1 by 10+ points. Treat the prompt as part of the evaluator: version it, diff it, and re-run the full evaluation set when it changes.
- **No calibrated probabilities.** An LLM that outputs `REFUND` doesn't tell you it's 0.83 confident. You can ask the model to emit a confidence score, but that score is itself a generated token, not a true posterior. Threshold-tuning tricks from classical ML don't transfer directly. Use abstain-and-route patterns instead: if the model emits low confidence or refuses to commit, escalate to a human reviewer.
- **Calibration drift.** The same prompt against the same model can produce different distributions of labels three months apart, because the model changed under the same alias. Pin the model id (`global.anthropic.claude-sonnet-4-6`, not `claude-sonnet`) and re-run evaluation when you upgrade. Calibration drift is one type of production drift; the [Curating your golden dataset](#curating-your-golden-dataset-over-time) section below covers the broader continuous-curation discipline.

For calibrating an LLM judge against human judgment, the same correlation-based pattern from [`subjective/human-alignment/`](../../subjective/human-alignment/README.md) applies. Even though classification is an objective task, the *labels themselves* may have been produced subjectively, and that subjectivity bleeds into your metrics.

## Ground truth quality is a prerequisite

Before you trust any metric, you have to trust your labels. Even objective metrics depend on label quality, and labels are often produced subjectively, by a human annotator, a heuristic, or a previous LLM. This is the most common failure mode for LLM classification evaluation in production.

A useful diagnostic is **Cohen's kappa** between two annotators on the same samples. Kappa measures agreement above chance. A kappa of 1.0 means perfect agreement, 0.0 means no better than random, and below 0.6 means the labels themselves are too noisy to trust. If your two human annotators only agree 60% of the time on what counts as `REFUND` vs `CANCEL`, no LLM will do better, and any metric you compute against either annotator's labels is misleading.

In the Bitext example above, we ran a 50-sample spot check and computed kappa across three pairs of annotators: Claude Sonnet, Claude Haiku, and the original Bitext labels. The Sonnet–Haiku kappa was 0.86, while the Bitext-original–Sonnet kappa was 0.74. Two LLMs agreed with each other more than either agreed with the dataset labels: a strong signal that the dataset itself has label noise on the class boundaries. The fix is not "tune the LLM more"; it's "audit the labels."

![Cohen's kappa across three annotator pairs: Sonnet-Haiku 0.86, Bitext-Haiku 0.70, Bitext-Sonnet 0.74](./figures/fig3_kappa_comparison.png)

The broader calibration loop, how to verify that an automated scorer agrees with a human expert, is covered in [`subjective/human-alignment/`](../../subjective/human-alignment/README.md). That pattern is built for LLM-as-judge, but the underlying idea applies here: your labels are an evaluator, and you have to verify that they actually match what a domain expert would say before you trust any metric computed against them.

## Curating your golden dataset over time

A golden evaluation dataset is not a one-time deliverable. The first version you ship to staging captures a slice of the world frozen in time. Production traffic shifts. New customer behaviors emerge. The model alias you depend on changes underneath you. Without a dataset that grows and adapts with reality, your evaluation metrics become a sticker on the wall: technically valid, no longer informative.

The discipline that closes this gap is **continuous curation**: a small, repeatable workflow that takes signals from production and feeds them back into the evaluation set on a regular cadence. Treat it as part of the system, not a side project.

### What goes into the golden set

A healthy golden set has three layers:

1. **Stable backbone** (60–70% of samples). The core test cases that define what "the agent works" means. These do not change month to month. They are the regression suite.
2. **Stratified production sample** (20–30% of samples). A periodically refreshed sample drawn from real production traffic, stratified by class so that rare classes are not under-sampled. This layer captures distributional drift over time.
3. **Hard cases and known failures** (5–15% of samples). Every misclassification you identify in production, every adversarial input a customer threw at the agent, every edge case a domain expert flagged. This layer prevents regressions on the things that have already burned you once.

When the size of any one layer falls outside its band, rebalance.

### A weekly curation loop

A practical cadence for most teams is weekly. The loop has four steps:

1. **Sample.** Pull a fixed number of new production tickets at random, stratified by predicted class. 50–200 per week is a typical range, scaled by traffic.
2. **Re-label.** Have a domain expert (or a second LLM as an initial filter, with kappa-validated agreement) label the sample. Compare against the agent's prediction.
3. **Triage.** Classify each disagreement as one of: (a) the agent was wrong and the label is correct, so the sample joins the "hard cases" layer; (b) the agent was right and the original label was wrong, so fix the label; (c) both are defensible, so review the class definition with the domain expert.
4. **Promote.** A subset of the triaged samples graduates into the next version of the golden set. Tag each promoted sample with its source week so drift across versions is auditable.

### Drift signals that should trigger an out-of-cycle update

- **Per-class F1 drops by more than 0.05 on the stable backbone between two evaluation runs** with no prompt or model change. Strong signal that production distribution has shifted in a way the backbone no longer captures.
- **Inter-annotator kappa on a fresh sample drops below 0.7** when historically it was above 0.85. The class boundaries have likely become ambiguous in the new traffic, and re-deriving the labeling guide should precede further metric tuning.
- **A new failure mode appears in customer escalations that is not represented in the golden set at all.** Add a small batch of these cases immediately, even outside the weekly cadence.

### Common anti-patterns

- **Letting the golden set grow without bound.** A 50,000-sample evaluation set is expensive to run on every prompt change. Cap the total size and rotate older samples into an archive.
- **Re-labeling with the same LLM that produced the predictions.** If the labeler and the classifier are both Sonnet, kappa will look great and tell you nothing. Use a different model family or a human reviewer.
- **Treating production labels as ground truth without sampling.** Production "ground truth" often comes from downstream signals (the customer clicked refund, the agent escalated, the ticket reopened) that are correlated with the agent's prediction in subtle ways. Always sample-and-relabel a portion to keep that bias visible.

## A self-serve checklist

Before you ship a single-label classifier to production:

1. ☐ Compute the class distribution. If max:min ratio > 2:1, do not rely on accuracy alone.
2. ☐ Report per-class precision, recall, and F1, not just the aggregate.
3. ☐ Look at the confusion matrix. Identify the top 3 confused class pairs.
4. ☐ Spot-check 30–50 samples with a second annotator. Compute kappa. If kappa < 0.7, fix the labels before tuning the model.
5. ☐ For every class with F1 < 0.7, read 5 misclassifications and attribute the failure: prompt ambiguity, label noise, or genuine model limitation.
6. ☐ Set per-class confidence thresholds where the cost of error is asymmetric (e.g. abstain on low-confidence `REFUND` predictions and route to a human).
7. ☐ Set up a weekly curation loop (sample, re-label, triage, promote). Even 50 samples a week prevents drift from becoming invisible.
8. ☐ Cap the golden set size and rotate. An evaluation set you cannot afford to run on every prompt change is not actually "golden".
9. ☐ Re-evaluate when the model alias changes. If you depend on `claude-sonnet`, pin to `global.anthropic.claude-sonnet-4-6` (or whatever specific snapshot is current) and re-run the full set on each upgrade.

## When to escalate to human annotation

- Inter-annotator kappa < 0.6 → your class definitions are ambiguous; rewrite the labeling guide before continuing.
- A specific class consistently scores F1 < 0.5 across multiple prompt iterations → the class may be inherently hard or the boundary with a neighboring class may be wrong; consider merging classes or adding a domain expert review step in production.
- Small prompt changes cause large metric swings → your label set is noisy enough that the model is fitting noise; spot-check more samples and re-derive ground truth.
- Per-class F1 drops by more than 0.05 on the stable backbone between consecutive evaluation runs, with no prompt or model change → production distribution has likely shifted; re-sample and re-label before adjusting the agent.
