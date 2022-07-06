from torch.utils.data import Dataset
from .pipelines import Compose


class BaseDataset(Dataset):
    def __init__(
        self,
        pipeline,
        data_root=None,
        test_mode=False,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.test_mode = test_mode

        self.pipeline = Compose(pipeline)

    def load_img_files(self):
        raise NotImplementedError

    def load_labels(self):
        raise NotImplementedError

    def prepare_train_img(self, index):
        raise NotImplementedError

    def prepare_test_img(self, index):
        raise NotImplementedError

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        return self.prepare_train_img(idx)
