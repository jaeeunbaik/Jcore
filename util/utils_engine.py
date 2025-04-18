from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, patience=5, verbose=False):
        super().__init__(monitor='val_loss', patience=patience, verbose=verbose)

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath='checkpoints/{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', save_top_k=1):
        super().__init__(filepath=filepath, monitor=monitor, save_top_k=save_top_k)