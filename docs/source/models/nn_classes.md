# NN implementation

While RealMLP is implemented in PyTorch, 
we extend the conventional `nn.Module` logic. 
Traditionally, one writes some PyTorch code to assemble a NN model, 
which is a nn.Module composed of building blocks 
that are also nn.Module objects (Composite design pattern). 
The nn.Module classes initialize the parameters in the constructor 
and are then callable objects providing the forward() transformation. 
Data preprocessing is done separately via different code/classes. 
We use a different structure of classes that unifies preprocessing and NN layers, 
which is useful for vectorized NNs: 
The vectorized NNs can share a single non-preprocessed data set, 
loaded into GPU RAM, 
while having different preprocessing parameters 
(fitted on different training sets since different splits are used). 
Individual preprocessed data sets are never fully instantiated in GPU RAM; 
instead, the vectorized NN models do preprocessing on batches individually, 
which saves GPU RAM (we're talking e.g. 
about having 50-100 NNs on the same GPU at the same time).
The class structure uses three base classes:

- `Layer` classes are similar to nn.Module, 
but they do not perform random initialization in the constructor. 
Instead, they simply take the already initialized parameters as input. 
There are some additional features: 
Layer objects of the same type can be combined into a vectorized Layer. 
The vectorized NN is not built directly, 
but first NNs are built and initialized sequentially for better reproducibility 
(random seed etc.) and RAM saving, 
and then they are vectorized after initialization using the Layer.stack() function. 
Additionally, Layer classes work with the DictDataset class, 
which usually contains 'x_cont' and 'x_cat' tensors 
for continuous and categorical variables. 
Moreover, during training, we also pass the labels 'y' through the Layer, 
which allows to implement mixup, label smoothing, 
and output standardization as Layer objects.
- `Fitter` classes initialize the NN based on a single forward pass 
on the (subsampled) training (and possibly validation) set. 
This is done using the `fit()` or `fit_transform()` functions 
similar to scikit-learn preprocessing classes, 
which return a `Layer` object 
(and, in case of `fit_transform()`, the transformed dataset). 
Initialization can be random or depending on the so far transformed training set. 
Typically, parameters of preprocessing layers 
such as standardization depend on the training set, 
while NN parameters do not depend on the training set. 
However, we also use weight and bias initializations 
that depend on the training set, 
and the unification of NN and preprocessing makes this much more convenient. 
- `FitterFactory` (could also be called ArchitectureBuilder) classes 
build the NN structure based on the input and output shape and type. 
Specifically, `FitterFactory` objects can build `Fitter` objects 
given the corresponding 'tensor_infos' of the data set, 
which specifies the number of continuous variables, 
the number of categorical variables and the category sizes, 
and the same for the labels. 
For example, a `FitterFactory` can decide to use one-hot encoding 
for categorical variables with small category sizes, 
and Embedding layers for larger category sizes.

The `Layer`, `Fitter`, and `FitterFactory` classes are defined in `model/base.py`. 
Other subclasses are also defined in `model` folder. There are some more features:

- We introduce a class called `Variable` that inherits from `torch.nn.Parameter`. 
Variable has a parameter `trainable: bool`, and in the case `trainable==False`, 
the `Layer` class will register it using `register_buffer()`. 
One might also be able to just use `nn.Parameter(..., requires_grad=False)`
for this, though we did not check whether it has the same effect 
(will it be saved when using `model.state_dict()`?). 
There is also the convenience function `Variable.stack()` used by `Layer.stack()`. 
Moreover, Variables can have names 
(to assign individual hyperparameter values to them), 
and they can have custom hyperparameter factors 
(e.g. to specify that the lr should be multiplied 
by a certain value for this Variable).
- The classes above can be given scope names, 
which are then prepended to variable names. 
For example, using scope names, 
the weight of the first linear layer 
in a NN could be called 'net/first_layer/layer-0/weight', 
where 0 is the layer index and 'first_layer' is 
redundant information that can be useful when regex matching variable names. 
One can assign an individual lr to this layer by using
`lr={'': global_lr, '.*first_layer.*weight': first_layer_weight_lr}`
in `**kwargs` to the `NNAlgInterface`. 
This works as follows: The `HyperparamManager`, 
which is available through a global context managed by the `TrainContext` class, 
stores the hyperparameter configurations obtained through **kwargs. 
Different classes can require getters for specific hyperparameters 
for specific variables. 
If multiple lr values are specified above, 
the one from the last matching regex is taken.
The scope names are passed on from FitterFactory to Fitter and then 
to Layer and Variable by a somewhat complicated context manager system, 
for which I didn't find a more elegant solution.
- Fitter objects can be split up in three parts using the `split_off_dynamic()` 
and `split_off_individual()` functions.
The static part would typically be the one-hot encoding, 
since it does not depend on the data and is not trainable, 
which means that even in a vectorized context, 
it can be applied once to the single shared data set 
since it does not depend on the train/val/test split.  
Then, there is the dynamic but not individual part, 
which can depend on the fitting data but is not trained or randomized, 
and can therefore be shared by models with the same trainval-test split. 
Finally, there is the individual (trainable/randomized) part, 
which is usually the NN part.
- `Fitter` classes should implement methods that allow to estimate 
the RAM usage of the parameters and a forward pass, 
which allows to decide how many NNs fit onto a GPU when running the benchmark.