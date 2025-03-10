# (Multi)quantile regression with RealMLP

RealMLP supports multiquantile regression, for example by using
```python
from pytabkit import RealMLP_TD_Regressor
reg = RealMLP_TD_Regressor(
    train_metric_name='multi_pinball(0.25,0.5,0.75)',
    val_metric_name='multi_pinball(0.25,0.5,0.75)'
)
```
This will adjust the training objective 
as well as the metric for best-epoch selection on the validation set.
The quantiles can be specified in any format 
that Python can convert to a float. 
There must be no space between the commas, 
and the quantiles need to be in ascending order.
The latter is relevant because RealMLP 
will by default sort the prediction outputs, 
to always have ascending quantile predictions.
This can be deactivated by passing `sort_quantile_predictions=False`.
