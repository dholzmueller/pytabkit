Scikit-learn interfaces
=======================

We provide scikit-learn interfaces for numerous methods in
``pytabkit.models.sklearn.sklearn_interfaces``.
Below, we provide an overview.
All of our interfaces allow to specify the validation set(s)
and categorical features in the ``fit`` method:

.. autofunction:: pytabkit.models.sklearn.sklearn_base.AlgInterfaceEstimator.fit


RealMLP
-------

For RealMLP, we provide TD (tuned default)
and HPO (hyperparameter optimization with random search) variants:

- RealMLP_TD_Classifier
- RealMLP_TD_Regressor
- RealMLP_HPO_Classifier
- RealMLP_HPO_Regressor

While the TD variants have good defaults,
they provide the option to override any hyperparameters.
The classifier and regressor have the same hyperparameters,
therefore we only show the constructor of the classifier here.
The first parameters until (including) verbosity
are provided for every scikit-learn interface,
although ``random_state``, ``n_threads``, ``tmp_folder``,
and ``verbosity`` may be ignored by some of the methods.

.. autofunction:: pytabkit.models.sklearn.sklearn_interfaces.RealMLP_TD_Classifier.__init__

For the HPO variants, we currently only provide few options:

.. autofunction:: pytabkit.models.sklearn.sklearn_interfaces.RealMLP_HPO_Classifier.__init__


Boosted Trees
-------------

For boosted trees, we provide the same interfaces as for RealMLP (TD, D, and HPO variants),
but do not wrap the full parameter space from the respective libraries.
Here are some representative examples:

.. autofunction:: pytabkit.models.sklearn.sklearn_interfaces.XGB_TD_Classifier.__init__
.. autofunction:: pytabkit.models.sklearn.sklearn_interfaces.LGBM_TD_Classifier.__init__
.. autofunction:: pytabkit.models.sklearn.sklearn_interfaces.CatBoost_TD_Classifier.__init__

Other NN baselines
---------

We offer interfaces (D and HPO variants) for

- MLP (from the RTDL code)
- ResNet (from the RTDL code)
- FTT (FT-Transformer from the RTDL code)
- MLP-PLR (from the RTDL code)
- TabR (requires installing faiss)
- TabM

.. autofunction:: pytabkit.models.sklearn.sklearn_interfaces.MLP_RTDL_D_Classifier.__init__
.. autofunction:: pytabkit.models.sklearn.sklearn_interfaces.Resnet_RTDL_D_Classifier.__init__
.. autofunction:: pytabkit.models.sklearn.sklearn_interfaces.FTT_D_Classifier.__init__
.. autofunction:: pytabkit.models.sklearn.sklearn_interfaces.MLP_PLR_D_Classifier.__init__
.. autofunction:: pytabkit.models.sklearn.sklearn_interfaces.TabR_S_D_Classifier.__init__
.. autofunction:: pytabkit.models.sklearn.sklearn_interfaces.TabM_D_Classifier.__init__

Other methods
-------------
For convenience, we wrap the scikit-learn RF and MLP interfaces
with our scikit-learn interfaces,
although in this case the validation sets are not used.
The respective classes are called
``RF_SKL_Classifier`` and ``MLP_SKL_Classifier`` etc.
We also provide our ``Ensemble_TD_Classifier``,
a weighted ensemble of our TD models (and similar for regression).

..
    test

    .. autoclass:: pytabkit.models.sklearn.sklearn_interfaces.RealMLPConstructorMixin

    test2

    .. automodule:: pytabkit.models.sklearn.sklearn_interfaces
        :members:
        :undoc-members:
        :show-inheritance:

Saving and loading
------------------

RealMLP and possibly other models (except probably TabR)
can be saved using pickle-like modules.
With standard pickling,
a model trained on a GPU will be restored to use the same GPU,
and fail to load if the GPU is not present.

The following code allows to load GPU-trained models to the CPU,
but fails to run predict() due to pytorch-lightning device issues.

.. code-block:: language
    import torch
    import dill  # might also work with pickle instad
    torch.save(model, 'model.pkl', pickle_module=dill, _use_new_zipfile_serialization=False)
    model = torch.load('model.pkl', map_location='cpu', pickle_module=dill)

