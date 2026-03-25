from datasets import load_dataset
from comptra.languages import *


def get_datasets(dataset_name_or_path: str, language: str):
    """
    Get a dataset given its description and the language of interest
    Arguments
    ---------
        - dataset_name_or_path: str,
            Description of the dataset of interest
        - language: str,
            Language of interest (e.g. English)
    Examples
    --------
    >>> get_datasets("flores", "English")
    DatasetDict({
        dev: Dataset({
            features: ['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink', 'sentence'],
            num_rows: 997
        })
        devtest: Dataset({
            features: ['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink', 'sentence'],
            num_rows: 1012
        })
    })
    """
    if dataset_name_or_path == "flores":
        if language in NON_FLORES:
            from comptra.data.extension import get_datasets as get_extension_datasets

            dataset = get_extension_datasets(MAPPING_LANG_TO_KEY[language])
        else:
            dataset = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])
    elif dataset_name_or_path == "ntrex":
        from comptra.data.ntrex import get_datasets as ntrex

        code = MAPPING_LANG_TO_KEY_NTREX[language]
        dataset = ntrex(code, code)[0]
    elif dataset_name_or_path == "tico":
        from comptra.data.tico import get_datasets as tico

        if language == "English":
            # dataset, _ = tico("English", "Hausa")
            dataset, _ = tico("English", "Bengali")
        else:
            _, dataset = tico("English", language)
    elif dataset_name_or_path == "ood":
        # dev = Flores, devtest = TICO
        from comptra.data.tico import get_datasets as tico

        # FLORES-200
        if language in NON_FLORES:
            from comptra.data.extension import get_datasets as get_extension_datasets

            flores_dataset = get_extension_datasets(MAPPING_LANG_TO_KEY[language])
        else:
            flores_dataset = load_dataset(
                "facebook/flores", MAPPING_LANG_TO_KEY[language]
            )
        # TICO-19
        if language == "English":
            # dataset, _ = tico("English", "Hausa")
            dataset, _ = tico("English", "Bengali")
        else:
            _, dataset = tico("English", language)
        # dev = Flores, devtest = TICO
        dataset["dev"] = flores_dataset["dev"]
    elif dataset_name_or_path == "validation":
        if language in NON_FLORES:
            from comptra.data.extension import get_datasets as get_extension_datasets

            dataset = get_extension_datasets(MAPPING_LANG_TO_KEY[language])
        else:
            dataset = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])
        from datasets import DatasetDict, Dataset

        return DatasetDict(
            {
                "devtest": dataset["dev"],
                "dev": Dataset.from_dict(
                    {col: [] for col in dataset["dev"].column_names}
                ),
            }
        )
    elif dataset_name_or_path == "topxgen":
        from train_datasets import get
        from datasets import Dataset

        dataset = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])
        A = load_dataset("almanach/topxgen-gemma-3-27b-and-nllb-3.3b", split=language)
        dataset_src = load_dataset("facebook/flores", "eng_Latn")
        dataset_src["devtest"] = dataset_src["devtest"].remove_columns(
            ["id", "URL", "domain", "topic", "has_image", "has_hyperlink"]
        )
        dataset_src["dev"] = Dataset.from_dict({"sentence": A["source"]})

        dataset["devtest"] = dataset["devtest"].remove_columns(
            ["id", "URL", "domain", "topic", "has_image", "has_hyperlink"]
        )
        dataset["dev"] = Dataset.from_dict({"sentence": A["target"]})
        return dataset_src, dataset
    elif dataset_name_or_path == "wmt24":
        from datasets import DatasetDict, Dataset
        REMOVE = [0, 167, 265, 309, 312, 474, 504, 532, 545, 605, 612, 613, 257, 262, 267, 288, 293, 387, 405, 436, 438, 445, 447, 449, 451, 657, 658, 660, 661, 662, 583, 593, 426, 435, 594, 595, 659, 663]
        if language == "English":
            devtest = load_dataset("google/wmt24pp", "en-is_IS")
        else:
            devtest = load_dataset(
                "google/wmt24pp", f"en-{CODE_BY_LANGUAGE[language]}"
            )
        dev = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])["dev"]
        dataset = DatasetDict(
            {
                "devtest": Dataset.from_dict(
                    {
                        "sentence": (
                            devtest["train"]["source"]
                            if language == "English"
                            else devtest["train"]["target"]
                        )
                    }
                ),
                "dev": dev,
            }
        )
        dataset["devtest"] = dataset["devtest"].filter(lambda example, idx: idx not in REMOVE, with_indices=True)
        return dataset
    else:
        raise ValueError(f"Unsupported dataset description '{dataset_name_or_path}")
    return dataset
