import os

import kaggle

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')


def main():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_file(
        dataset='nelgiriyewithana/credit-card-fraud-detection-dataset-2023',
        file_name='creditcard_2023.csv',
        path=DATA_DIR,
    )


if __name__ == '__main__':
    main()
