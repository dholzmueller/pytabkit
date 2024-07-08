# Adding your own models to the benchmark

To run your own models,
- implement an `AlgInterface` subclass. There are numerous examples already implemented.
For models that can only run a single train-validation-test split at a time, 
you might want to subclass or modify `SklearnSubSplitInterface` from 
`pytabkit/models/alg_interfaces/sub_split_interfaces.py`. Examples can be found in
`pytabkit/models/alg_interfaces/other_interfaces.py` or
`pytabkit/models/alg_interfaces/rtdl_interfaces.py`.
- add an `AlgInterfaceWrapper` subclass. This is often just a three-liner 
that specifies which AlgInterfaces subclass to instantiate. 
See the numerous examples in
`pytabkit/bench/alg_wrappers/interface_wrappers.py`, especially the later ones.
- adjust the code to run your `AlgInterfaceWrapper` on the benchmark,
see `scripts/run_experiments.py` for many examples. 
Note that `RunConfig` has an option to save the model predictions 
on the whole datasets,
which can significantly increase the disk usage 
(can be up to 2 GB per model on the meta-test-class benchmark).