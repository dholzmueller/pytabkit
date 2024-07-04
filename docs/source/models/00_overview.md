# Overview of the `models` part

## Scikit-learn interfaces

We provide scikit-learn interfaces for various methods in 
`sklearn/sklearn_interfaces.py`. 
These use the default parameter dictionaries defined in `sklearn/default_params.py`.

## AlgInterface: more fine-grained control

We implement all our methods 
through subclassing `AlgInterface` in `alg_interfaces/alg_interfaces.py`.
`AlgInterface` provides more functionality than scikit-learn interfaces,
which is crucial for our benchmarking in `pytabkit.bench`. 
All our scikit-learn interfaces are wrappers around `AlgInterface` classes, 
using the `sklearn.sklearn_base.AlgInterfaceEstimator` base class.
Compared to scikit-learn interfaces, 
`AlgInterface` provides the following additional features:

- Vectorized evaluation on multiple train-validation-test splits 
(used by RealMLP-TD and RealMLP-TD-S).
- Specification of train-validation-test splits, random seeds, temporary folder, custom loggers
- Inclusion of required resource estimates (CPU RAM, GPU RAM, GPU usage, n_threads, time)
- Evaluation on a list of metrics
- Refitting with best found parameters

## Hyperparameter handling

Hyperparameters are explicitly defined in scikit-learn constructors.
<!--- but otherwise, they are just passed on through `**kwargs` (often called `**config`).
Hence, some default parameters are just filled in 
at the point where the parameter is used.-->
Elsewhere, we generally pass all configuration parameters as **kwargs, 
then the corresponding functions pick out the parameters that they need 
and pass the rest on to nested function calls. 
This allows for very convenient coding, 
but one has to pay attention for typos in parameter names, 
which will often not be caught. 
For example, one could have the following structure:

```python

def fit(**kwargs):
    model = build_model(**kwargs)
    train_model(model, **kwargs)
    
def build_model(n_layers=4, **kwargs):
    ...
    
def train_model(model, lr=4e-2, batch_size=256, **kwargs):
    ...
```
    
We usually write `**config` instead of `**kwargs`. 
We also generally try to give unique names to parameters. 
For example, the epsilon parameter of the optimizer 
is called `opt_eps` and the epsilon parameter of label smoothing is called `ls_eps`.

## Internal data representation

We represent datasets internally using the `DictDataset` class. 
It contains a dictionary of PyTorch tensors. 
In our case, there are usually three tensors: 
`'x_cont'` for continuous features,
`'x_cat'` for categorical features (`dtype=torch.long`), and
`'y'` for labels.
A `DictDataset` also contains a dictionary `tensor_infos`, 
which for each of these keys contains a `TensorInfo` object. 
The latter describes the number of features and, 
if applicable, the number of categories for each feature
(for categorical variables or classification labels).

We reserve the category `0` as the category for missing values 
(and values that have not been known to exist at train time).
Missing numerical values are currently not handled by the NN code, 
so they need to be encoded beforehand.


## Data preprocessing (also available for other models)

Most models offer to customize the data preprocessing 
through the `tfms` parameter. 
This is done using the NN preprocessing code in 
`nn_models.models.PreprocessingFactory`
(see the corresponding documentation page 
for an explanation of the Factory classes).

## NN implementation

For the implementation of RealMLP, 
we extend and alter the typical PyTorch structure, 
see the documentation page on NN classes.

## Vectorization

Due to the vectorization of NN models, we use different terms for similar things:
- `n_cv` refers to the number 
of training-validation splits in cross-validation (bagging)
- `n_refit` refers to the number of models 
that are refitted on training+validation data after the CV stage
- `n_tv_splits` (or `n_models`) refers to the number of training-validation 
splits used in the current training (could be `n_cv` or `n_refit`)
- `n_tt_splits` (or `n_parallel`) refers to the number of trainval-test splits used
(this is normally 1 when used through the scikit-learn interface,
but can be larger when using RealMLP through the benchmark)
