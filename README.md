# Scalable Linear Regression with PyTorch Lightning

This linear regression model allows you to scale to much bigger datasets by having the option to train on multiple GPUS and TPUS. I implemented this model in the [PyTorch Lightning Bolts](https://github.com/PyTorchLightning/pytorch-lightning-bolts) library, where it has been rigorously tested and documented.

I've also implemented the SklearnDataModule - a class that conveniently puts any Numpy array dataset into PyTorch DataLoaders.

Check out the Bolts documentation if you have any questions about how to use this model
* [Linear Regression docs](https://pytorch-lightning-bolts.readthedocs.io/en/latest/classic_ml.html#linear-regression)
* [Sklearn DataModule docs](https://pytorch-lightning-bolts.readthedocs.io/en/latest/sklearn_datamodule.html)

## An example

Train this model on any Numpy dataset as follows (here I'm demonstrating with the Sklearn Boston dataset):

```python
from pl_bolts.models.regression import LinearRegression
from pl_bolts.datamodules import SklearnDataModule
import pytorch_lightning as pl

# use any numpy or sklearn dataset
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)
loaders = SklearnDataModule(X, y)

# build model
model = LinearRegression(input_dim=13)

# fit
trainer = pl.Trainer()
trainer.fit(model, loaders.train_dataloader(), loaders.val_dataloader())

# test
trainer.test(test_dataloaders=loaders.test_dataloader())
```

To specify the number of GPUs or TPUs, just specify the flag in the Trainer. You can also enable 16-bit precision in the Trainer.
```python
# 1 GPU
trainer = pl.Trainer(gpus=1)

# 8 TPUs
trainer = pl.Trainer(tpu_cores=8)

# 16 GPUs and 16-bit precision
trainer = pl.Trainer(gpus=16, precision=16)
```
