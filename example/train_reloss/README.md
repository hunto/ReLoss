## Learning ReLoss

Taking image classification (ImageNet) as an example.

* Install [diffsort](https://github.com/Felix-Petersen/diffsort)

    ```shell
    pip install diffsort
    ```

* Prepare model predictions and labels

    Train a model (we use ResNet-50 in the paper) using CELoss from scratch, and write additional code to store the predicted logits and labels on ImageNet training set during training.

* Train reloss

    Modify the code in `train_reloss.py` to load the stored predictions and labels. Then run

    ```shell
    python train_reloss.py
    ```

The best checkpoint would be stored in `./loss_module_best.ckpt` .