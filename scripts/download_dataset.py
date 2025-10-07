import os

import kaggle

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')


def main():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset='mlg-ulb/creditcardfraud',
        path=DATA_DIR,
        unzip=True,
    )


if __name__ == '__main__':
    main()
