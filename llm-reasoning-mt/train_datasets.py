import os
import json
import numpy as np
from typing import List, Union
from datasets import Dataset, concatenate_datasets, load_dataset
from comptra.languages import MAPPING_LANG_TO_KEY


def get_flores(
    src: str,
    languages: List[str],
    test_size_ratio: Union[float, int],
    seed: int,
    size: int = None,
):
    list_of_datasets = []
    ds_src = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[src])
    for language in languages:
        ds_tgt = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])
        dataset = Dataset.from_dict(
            {
                "source": ds_src["dev"]["sentence"],
                "target": ds_tgt["dev"]["sentence"],
                "source_language": [src] * len(ds_src["dev"]),
                "target_language": [language] * len(ds_src["dev"]),
            }
        )
        # dataset = dataset.rename_column("source", input_column_name)
        # dataset = dataset.rename_column("target", output_column_name)
        if size is not None and size > 0:
            dataset = dataset.select([i for i in range(size)])
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)
    ds = dataset.train_test_split(test_size=test_size_ratio, shuffle=True, seed=seed)
    return ds


MP = {
    "Hausa": "ha",
    "Igbo": "ig",
    "Kinyarwanda": "rw",
    "Somali": "so",
    "Swahili": "sw",
    "Xhosa": "xh",
}


def get_smol(
    src: str, languages: List[str], test_size_ratio: Union[float, int], seed: int
):
    assert src == "English"
    list_of_datasets = []
    for language in languages:
        ds_src = load_dataset("google/smol", f"smolsent__en_{MP[language]}")
        dataset = Dataset.from_dict(
            {
                "source": ds_src["train"]["src"],
                "target": ds_src["train"]["trg"],
                "source_language": [src] * len(ds_src["train"]),
                "target_language": [language] * len(ds_src["train"]),
            }
        )
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)
    ds = dataset.train_test_split(test_size=test_size_ratio, shuffle=True, seed=seed)
    return ds


def get_paraphrase(
    data_dir: str,
    languages: List[str],
    size: int,
    test_size_ratio: Union[float, int],
    seed: int,
    reverse: bool = False,
    suffix: str = None,
):
    rng = np.random.default_rng(seed)
    list_of_datasets = []
    for language in languages:
        list_of_sentences = []
        list_of_translations = []
        list_of_demos = []
        list_of_reverse_demos = []
        if suffix is None:
            filename = f"{language}_paraphrase_comptra.jsonl"
        else:
            filename = f"{language}_paraphrase_{suffix}.jsonl"
        with open(os.path.join(data_dir, filename), "r") as fin:
            for line in fin:
                dico = json.loads(line)
                sentence = dico["sentence"]
                translation = dico["translation"]
                paraphrases = dico["paraphrases"]
                paraphrases_translations = dico["translations"]
                if len(sentence.strip()) < 10 or len(translation.strip()) < 10:
                    continue
                if any([col in translation for col in ["#", ">"]]) or any(
                    [col in sentence for col in ["#", ">"]]
                ):
                    continue
                if (
                    len(paraphrases) == 0
                    or len(paraphrases_translations) == 0
                    or len(paraphrases) != len(paraphrases_translations)
                ):
                    continue
                # """
                demos = "<Demonstrations>\n"
                reverse_demos = "<Demonstrations>\n"
                for j, (s, t) in enumerate(zip(paraphrases, paraphrases_translations)):
                    demos += (
                        f"{j+1}. English sentence\n{s}\n{language} translation\n{t}\n\n"
                    )
                    reverse_demos += (
                        f"{j+1}. {language} sentence\n{t}\nEnglish translation\n{s}\n\n"
                    )

                    # demos += f"{j+1}. English paraphrase\n{s}\n\n"
                    # reverse_demos += f"{j+1}. {language} paraphrase\n{t}\n\n"

                    # demos += f"{j+1}. {language} paraphrase\n{t}\n\n"
                    # reverse_demos += f"{j+1}. English paraphrase\n{s}\n\n"
                demos = f"{demos.strip()}\n</Demonstrations>"
                reverse_demos = f"{reverse_demos.strip()}\n</Demonstrations>"
                list_of_sentences.append(sentence)
                list_of_translations.append(translation)
                list_of_demos.append(demos)
                list_of_reverse_demos.append(reverse_demos)
                # """
        if size < 0:
            selected_indices = [i for i in range(len(list_of_translations))]
        else:
            selected_indices = rng.choice(
                a=len(list_of_sentences), size=size, replace=False
            ).tolist()

        selected_indices = [i for i in selected_indices if i < len(list_of_sentences)]
        dataset = Dataset.from_dict(
            {
                "source": [list_of_sentences[i] for i in selected_indices],
                "target": [list_of_translations[i] for i in selected_indices],
                "source_language": ["English"] * len(selected_indices),
                "target_language": [language] * len(selected_indices),
                "demos": list_of_demos,
                "reverse_demos": list_of_demos,
            }
        )
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)
    if reverse:
        reverse_dataset = Dataset.from_dict(
            {
                "source": dataset["target"],
                "target": dataset["source"],
                "source_language": dataset["target_language"],
                "target_language": dataset["source_language"],
                "demos": dataset["reverse_demos"],
                "reverse_demos": dataset["demos"],
            }
        )

        print(
            f"We will consider both translation direction: English to {language} and {language} to English."
        )

        dataset = concatenate_datasets([dataset, reverse_dataset]).shuffle(seed=seed)

    def f(example):
        out = example["demos"] + f"\n\nFinal Translation\n{example['target']}"
        return {"new_target": out}

    updated_dataset = dataset.map(lambda x: f(x))
    updated_dataset = updated_dataset.remove_columns(
        ["demos", "reverse_demos", "target"]
    )
    updated_dataset = updated_dataset.rename_column("new_target", "target")
    if test_size_ratio == 0:
        return updated_dataset

    ds = updated_dataset.train_test_split(
        test_size=(1 + reverse) * test_size_ratio, shuffle=True, seed=seed
    )
    return ds


def get_paraphrase_2(  # Extension
    data_dir: str,
    languages: List[str],
    size: int,
    test_size_ratio: Union[float, int],
    seed: int,
    reverse: bool = False,
    suffix: str = None,
):
    rng = np.random.default_rng(seed)
    list_of_datasets = []
    for language in languages:
        if suffix is None:
            filename = f"{language}_paraphrase_comptra.jsonl"
        else:
            filename = f"{language}_paraphrase_{suffix}.jsonl"
        list_of_sentences = []
        list_of_translations = []
        with open(os.path.join(data_dir, filename), "r") as fin:
            for line in fin:
                dico = json.loads(line)
                sentence = dico["sentence"]
                translation = dico["translation"]
                paraphrases = dico["paraphrases"]
                paraphrases_translations = dico["translations"]
                if len(sentence.strip()) < 10 or len(translation.strip()) < 10:
                    continue
                if any([col in translation for col in ["#", ">"]]) or any(
                    [col in sentence for col in ["#", ">"]]
                ):
                    continue
                list_of_sentences.append(sentence)
                list_of_translations.append(translation)
                if (
                    len(paraphrases) == 0
                    or len(paraphrases_translations) == 0
                    or len(paraphrases) != len(paraphrases_translations)
                ):
                    continue
                list_of_sentences.extend(paraphrases)
                list_of_translations.extend(paraphrases_translations)
        if size < 0:
            selected_indices = [i for i in range(len(list_of_translations))]
        else:
            selected_indices = rng.choice(
                a=len(list_of_sentences), size=size, replace=False
            ).tolist()

        selected_indices = [i for i in selected_indices if i < len(list_of_sentences)]
        dataset = Dataset.from_dict(
            {
                "source": [list_of_sentences[i] for i in selected_indices],
                "target": [list_of_translations[i] for i in selected_indices],
                "source_language": ["English"] * len(selected_indices),
                "target_language": [language] * len(selected_indices),
            }
        )
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)
    if reverse:
        reverse_dataset = Dataset.from_dict(
            {
                "source": dataset["target"],
                "target": dataset["source"],
                "source_language": dataset["target_language"],
                "target_language": dataset["source_language"],
            }
        )

        print(
            f"We will consider both translation direction: English to {language} and {language} to English."
        )

        dataset = concatenate_datasets([dataset, reverse_dataset]).shuffle(seed=seed)

    if test_size_ratio == 0:
        return dataset

    ds = dataset.train_test_split(
        test_size=(1 + reverse) * test_size_ratio, shuffle=True, seed=seed
    )
    return ds


def get_sbys(
    data_dir: str,
    languages: List[str],
    size: int,
    test_size_ratio: Union[float, int],
    seed: int,
    reverse: bool = False,
):
    rng = np.random.default_rng(seed)
    list_of_datasets = []
    for language in languages:
        list_of_sentences = []
        list_of_translations = []
        list_of_sbys = []
        with open(
            os.path.join(data_dir, f"{language}_paraphrase_sbys.jsonl"), "r"
        ) as fin:
            for line in fin:
                dico = json.loads(line)
                sentence = dico["sentence"]
                translation = dico["translation"]
                if len(sentence.strip()) < 10 or len(translation.strip()) < 10:
                    continue
                if any([col in translation for col in ["#", ">"]]) or any(
                    [col in sentence for col in ["#", ">"]]
                ):
                    continue
                sbys = "<think>\n"
                sbys += dico["research"].strip()
                sbys += f"\n\n{dico['draft'].strip()}"
                sbys += "\n\nNow let's move to the next stage: Post-editing with local refinement."
                sbys += "\nIn this stage, the primary aim is to refine the draft translation by making micro-level improvements that improve the draft's fluency."
                sbys += f"\n\nHere is a refined version of the translation\n{dico['refinement']}"
                sbys += "\n\nNow, we will proofread the refined text for grammar spelling, punctuation, terminology and overall fluency."
                sbys += f"\n\nHere is the translation after proofreading\n{dico['proofreading']}"
                sbys += "\n\nWe will further improve it to obtain the final, polished translation."
                sbys = f"{sbys.strip()}\n</think>"
                list_of_sentences.append(sentence)
                list_of_translations.append(translation)
                list_of_sbys.append(sbys)
        if size < 0:
            selected_indices = [i for i in range(len(list_of_translations))]
        else:
            selected_indices = rng.choice(
                a=len(list_of_sentences), size=size, replace=False
            ).tolist()

        selected_indices = [i for i in selected_indices if i < len(list_of_sentences)]
        dataset = Dataset.from_dict(
            {
                "source": [list_of_sentences[i] for i in selected_indices],
                "target": [list_of_translations[i] for i in selected_indices],
                "source_language": ["English"] * len(selected_indices),
                "target_language": [language] * len(selected_indices),
                "sbys": list_of_sbys,
                # "reverse_sbys": list_of_reverse_sbys,
            }
        )
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)
    if reverse:
        reverse_dataset = Dataset.from_dict(
            {
                "source": dataset["target"],
                "target": dataset["source"],
                "source_language": dataset["target_language"],
                "target_language": dataset["source_language"],
                "sbys": dataset["reverse_sbys"],
                "reverse_sbys": dataset["sbys"],
            }
        )

        print(
            f"We will consider both translation direction: English to {language} and {language} to English."
        )

        dataset = concatenate_datasets([dataset, reverse_dataset]).shuffle(seed=seed)

    def f(example):
        out = example["sbys"] + f"\n\nFinal Translation\n{example['target']}"
        return {"new_target": out}

    updated_dataset = dataset.map(lambda x: f(x))
    # updated_dataset = updated_dataset.remove_columns(["sbys", "reverse_sbys", "target"])
    updated_dataset = updated_dataset.remove_columns(["sbys", "target"])
    updated_dataset = updated_dataset.rename_column("new_target", "target")
    if test_size_ratio == 0:
        return updated_dataset

    ds = updated_dataset.train_test_split(
        test_size=(1 + reverse) * test_size_ratio, shuffle=True, seed=seed
    )
    return ds


def get_maps(
    data_dir: str,
    languages: List[str],
    size: int,
    test_size_ratio: Union[float, int],
    seed: int,
    reverse: bool = False,
):
    triggers = [
        "services = iinkonzo\n",
        "AOL\n",
        "Microsoft\n",
        "services = iinkonzo \n",
        "AOL \n",
        "Microsoft \n",
    ]
    rng = np.random.default_rng(seed)
    list_of_datasets = []
    for language in languages:
        list_of_sentences = []
        list_of_translations = []
        list_of_maps = []
        with open(
            os.path.join(data_dir, f"{language}_paraphrase_maps.jsonl"), "r"
        ) as fin:
            for line in fin:
                dico = json.loads(line)
                sentence = dico["sentence"]
                translation = dico["translation"]
                # translation = dico["better-translation"]
                if len(sentence.strip()) < 10 or len(translation.strip()) < 10:
                    continue
                if any([col in translation for col in ["#", ">"]]) or any(
                    [col in sentence for col in ["#", ">"]]
                ):
                    continue
                demonstrations = dico["demonstrations"]
                keywords = dico["keywords"]
                start_keywords = 0
                for trigger in triggers:
                    if keywords.find(trigger) >= 0:
                        start_keywords = max(
                            start_keywords, keywords.find(trigger) + len(trigger)
                        )
                keywords = keywords[start_keywords:].strip()
                topics = dico["topics"]

                demos_trans = dico["demos-trans"]
                keywords_trans = dico["keywords-trans"]
                topics_trans = dico["topics-trans"]

                zs = dico["zero-shot"]
                article = (
                    "an"
                    if any(
                        [
                            language.startswith(vowel)
                            for vowel in ["a", "e", "i", "o", "u"]
                        ]
                    )
                    else "a"
                )
                maps = "<think>\n"
                maps += f"Here is a draft translation\n\n1. {zs}\n\n"
                maps += f"Let's write an English sentence related to but different from the input English sentence and translate it into {language}\n\n"
                maps += demonstrations + "\n\n"
                maps += f"Given this knowledge, we can draft another translation\n\n2. {demos_trans}\n\n"
                maps += f"Let's extract the keywords in the provided English sentence, and then translate these keywords into {language}\n\n"
                maps += keywords + "\n\n"
                maps += f"Given this knowledge, we can draft another translation\n\n3. {keywords_trans}\n\n"
                maps += f"Let's use a few words to describe the topics of the provided English sentence\n\n"
                maps += topics + "\n\n"
                maps += f"Given this knowledge, we can draft another translation\n\n4. {topics_trans}\n\n"
                maps += "We will choose the best of these translations and further improve it to obtain the final, polished translation."
                maps = f"{maps.strip()}\n</think>"
                list_of_sentences.append(sentence)
                list_of_translations.append(translation)
                list_of_maps.append(maps)
        if size < 0:
            selected_indices = [i for i in range(len(list_of_translations))]
        else:
            selected_indices = rng.choice(
                a=len(list_of_sentences), size=size, replace=False
            ).tolist()

        selected_indices = [i for i in selected_indices if i < len(list_of_sentences)]
        dataset = Dataset.from_dict(
            {
                "source": [list_of_sentences[i] for i in selected_indices],
                "target": [list_of_translations[i] for i in selected_indices],
                "source_language": ["English"] * len(selected_indices),
                "target_language": [language] * len(selected_indices),
                "maps": list_of_maps,
                # "reverse_maps": list_of_reverse_maps,
            }
        )
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)
    if reverse:
        reverse_dataset = Dataset.from_dict(
            {
                "source": dataset["target"],
                "target": dataset["source"],
                "source_language": dataset["target_language"],
                "target_language": dataset["source_language"],
                "maps": dataset["reverse_maps"],
                "reverse_maps": dataset["maps"],
            }
        )

        print(
            f"We will consider both translation direction: English to {language} and {language} to English."
        )

        dataset = concatenate_datasets([dataset, reverse_dataset]).shuffle(seed=seed)

    def f(example):
        out = example["maps"] + f"\n\nFinal Translation\n{example['target']}"
        return {"new_target": out}

    updated_dataset = dataset.map(lambda x: f(x))
    # updated_dataset = updated_dataset.remove_columns(["maps", "reverse_maps", "target"])
    updated_dataset = updated_dataset.remove_columns(["maps", "target"])
    updated_dataset = updated_dataset.rename_column("new_target", "target")
    if test_size_ratio == 0:
        return updated_dataset

    ds = updated_dataset.train_test_split(
        test_size=(1 + reverse) * test_size_ratio, shuffle=True, seed=seed
    )
    return ds


from comptra.utils import is_lang


def get_refine(
    data_dir: str,
    languages: List[str],
    size: int,
    test_size_ratio: Union[float, int],
    seed: int,
    reverse: bool = False,
):
    rng = np.random.default_rng(seed)
    list_of_datasets = []
    for language in languages:
        list_of_sentences = []
        list_of_translations = []
        list_of_demos = []
        with open(
            os.path.join(data_dir, f"{language}_paraphrase_refine.jsonl"), "r"
        ) as fin:
            for line in fin:
                dico = json.loads(line)
                sentence = dico["sentence"]
                translation = dico["translation"]
                if len(sentence.strip()) < 10 or len(translation.strip()) < 10:
                    continue
                if any([col in translation for col in ["#", ">"]]) or any(
                    [col in sentence for col in ["#", ">"]]
                ):
                    continue
                refined_outputs = dico["refined_outputs"]
                refined_outputs = [
                    element for element in refined_outputs if is_lang(element, language)
                ]
                if len(refined_outputs) < 4:
                    continue
                demos = "<think>\n"
                demos += f"Here is a draft translation\n\n1. {refined_outputs[0]}\n\n"
                demos += f"Let's improve it and write a better translation\n\n2. {refined_outputs[1]}\n\n"
                demos += f"Let's further improve it and write a better translation\n\n3. {refined_outputs[2]}\n\n"
                demos += f"Let's improve it one last time and write a better translation\n\n4. {refined_outputs[3]}\n\n"
                demos += "We will choose the best of these translations and further improve it to obtain the final, polished translation."
                demos = f"{demos.strip()}\n</think>"
                list_of_sentences.append(sentence)
                list_of_translations.append(translation)
                list_of_demos.append(demos)
        if size < 0:
            selected_indices = [i for i in range(len(list_of_translations))]
        else:
            selected_indices = rng.choice(
                a=len(list_of_sentences), size=size, replace=False
            ).tolist()

        selected_indices = [i for i in selected_indices if i < len(list_of_sentences)]
        dataset = Dataset.from_dict(
            {
                "source": [list_of_sentences[i] for i in selected_indices],
                "target": [list_of_translations[i] for i in selected_indices],
                "source_language": ["English"] * len(selected_indices),
                "target_language": [language] * len(selected_indices),
                "demos": list_of_demos,
                # "reverse_demos": list_of_reverse_demos,
            }
        )
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)
    if reverse:
        reverse_dataset = Dataset.from_dict(
            {
                "source": dataset["target"],
                "target": dataset["source"],
                "source_language": dataset["target_language"],
                "target_language": dataset["source_language"],
                "demos": dataset["reverse_demos"],
                "reverse_demos": dataset["demos"],
            }
        )

        print(
            f"We will consider both translation direction: English to {language} and {language} to English."
        )

        dataset = concatenate_datasets([dataset, reverse_dataset]).shuffle(seed=seed)

    def f(example):
        out = example["demos"] + f"\n\nFinal Translation\n{example['target']}"
        return {"new_target": out}

    updated_dataset = dataset.map(lambda x: f(x))
    # updated_dataset = updated_dataset.remove_columns(["demos", "reverse_demos", "target"])
    updated_dataset = updated_dataset.remove_columns(["demos", "target"])
    updated_dataset = updated_dataset.rename_column("new_target", "target")
    if test_size_ratio == 0:
        return updated_dataset

    ds = updated_dataset.train_test_split(
        test_size=(1 + reverse) * test_size_ratio, shuffle=True, seed=seed
    )
    return ds


def get_tear(
    data_dir: str,
    languages: List[str],
    size: int,
    test_size_ratio: Union[float, int],
    seed: int,
    reverse: bool = False,
):
    rng = np.random.default_rng(seed)
    list_of_datasets = []
    for language in languages:
        list_of_sentences = []
        list_of_translations = []
        list_of_demos = []
        with open(
            os.path.join(data_dir, f"{language}_paraphrase_tear.jsonl"), "r"
        ) as fin:
            for line in fin:
                dico = json.loads(line)
                sentence = dico["sentence"]
                translation = dico["translation"]
                if len(sentence.strip()) < 10 or len(translation.strip()) < 10:
                    continue
                if any([col in translation for col in ["#", ">"]]) or any(
                    [col in sentence for col in ["#", ">"]]
                ):
                    continue
                demos = "<think>\n"
                demos += f"Here is a draft translation\n\n1. {dico['draft']}\n\n"
                demos += f"Let's identify errors and assess the quality of the draft translation.\n"
                demos += "The categories of errors are accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.\n"
                demos += "Each error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technical errors but do not disrupt the flow or hinder comprehension.\n\n"
                demos += f"Here are the MQM annotations of the draft:\n{dico['estimation']}\n\n"
                demos += f"Upon reviewing the translation and error information, we can refine the draft and obtain a better translation\n\n2. {dico['refinement']}\n\n"
                demos += "We will further improve it to obtain the final, polished translation."
                demos = f"{demos.strip()}\n</think>"
                list_of_sentences.append(sentence)
                list_of_translations.append(translation)
                list_of_demos.append(demos)
        if size < 0:
            selected_indices = [i for i in range(len(list_of_translations))]
        else:
            selected_indices = rng.choice(
                a=len(list_of_sentences), size=size, replace=False
            ).tolist()

        selected_indices = [i for i in selected_indices if i < len(list_of_sentences)]
        dataset = Dataset.from_dict(
            {
                "source": [list_of_sentences[i] for i in selected_indices],
                "target": [list_of_translations[i] for i in selected_indices],
                "source_language": ["English"] * len(selected_indices),
                "target_language": [language] * len(selected_indices),
                "demos": list_of_demos,
                # "reverse_demos": list_of_reverse_demos,
            }
        )
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)
    if reverse:
        reverse_dataset = Dataset.from_dict(
            {
                "source": dataset["target"],
                "target": dataset["source"],
                "source_language": dataset["target_language"],
                "target_language": dataset["source_language"],
                "demos": dataset["reverse_demos"],
                "reverse_demos": dataset["demos"],
            }
        )

        print(
            f"We will consider both translation direction: English to {language} and {language} to English."
        )

        dataset = concatenate_datasets([dataset, reverse_dataset]).shuffle(seed=seed)

    def f(example):
        out = example["demos"] + f"\n\nFinal Translation\n{example['target']}"
        return {"new_target": out}

    updated_dataset = dataset.map(lambda x: f(x))
    # updated_dataset = updated_dataset.remove_columns(["demos", "reverse_demos", "target"])
    updated_dataset = updated_dataset.remove_columns(["demos", "target"])
    updated_dataset = updated_dataset.rename_column("new_target", "target")
    if test_size_ratio == 0:
        return updated_dataset

    ds = updated_dataset.train_test_split(
        test_size=(1 + reverse) * test_size_ratio, shuffle=True, seed=seed
    )
    return ds


def get_cot(
    data_dir: str,
    languages: List[str],
    size: int,
    test_size_ratio: Union[float, int],
    seed: int,
    reverse: bool = False,
    cot_template: int = 1,
):
    rng = np.random.default_rng(seed)
    list_of_datasets = []
    for language in languages:
        list_of_sentences = []
        list_of_translations = []
        list_of_cots = []
        with open(
            os.path.join(data_dir, f"{language}_paraphrase_cot_{cot_template}.jsonl"),
            "r",
        ) as fin:
            for line in fin:
                dico = json.loads(line)
                sentence = dico["sentence"]
                translation = dico["translation"]
                if len(sentence.strip()) < 10 or len(translation.strip()) < 10:
                    continue
                if any([col in translation for col in ["#", ">"]]) or any(
                    [col in sentence for col in ["#", ">"]]
                ):
                    continue
                cot = dico["chain_of_thought"]
                if "<think>" not in cot:
                    list_of_cots.append(f"<think>\n{cot}\n</think>")
                else:
                    idx_start = cot.find("<think>")
                    idx_end = cot.find("</think>")
                    if idx_start < 0 or idx_end < 0 or idx_start >= idx_end:
                        continue
                    if len(cot[idx_end + len("</think>") :].strip()) > 0:
                        # Everything after the </think> tag is considered as a CoT
                        cot = cot[idx_end + len("</think>") :].strip()
                    else:
                        # Everything between the <think> and </think> tags is considered as a CoT
                        cot = cot[idx_start + len("<think>") : idx_end].strip()
                    list_of_cots.append(f"<think>\n{cot}\n</think>")
                list_of_sentences.append(sentence)
                list_of_translations.append(translation)
        if size < 0:
            selected_indices = [i for i in range(len(list_of_translations))]
        else:
            selected_indices = rng.choice(
                a=len(list_of_sentences), size=size, replace=False
            ).tolist()

        selected_indices = [i for i in selected_indices if i < len(list_of_sentences)]
        dataset = Dataset.from_dict(
            {
                "source": [list_of_sentences[i] for i in selected_indices],
                "target": [list_of_translations[i] for i in selected_indices],
                "source_language": ["English"] * len(selected_indices),
                "target_language": [language] * len(selected_indices),
                "cot": list_of_cots,
                # "reverse_cot": list_of_reverse_cots,
            }
        )
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)
    if reverse:
        reverse_dataset = Dataset.from_dict(
            {
                "source": dataset["target"],
                "target": dataset["source"],
                "source_language": dataset["target_language"],
                "target_language": dataset["source_language"],
                "cot": dataset["reverse_cot"],
                "reverse_cot": dataset["cot"],
            }
        )

        print(
            f"We will consider both translation direction: English to {language} and {language} to English."
        )

        dataset = concatenate_datasets([dataset, reverse_dataset]).shuffle(seed=seed)

    def f(example):
        out = example["cot"] + f"\n\nFinal Translation\n{example['target']}"
        return {"new_target": out}

    updated_dataset = dataset.map(lambda x: f(x))
    # updated_dataset = updated_dataset.remove_columns(["cot", "reverse_cot", "target"])
    updated_dataset = updated_dataset.remove_columns(["cot", "target"])
    updated_dataset = updated_dataset.rename_column("new_target", "target")
    if test_size_ratio == 0:
        return updated_dataset

    ds = updated_dataset.train_test_split(
        test_size=(1 + reverse) * test_size_ratio, shuffle=True, seed=seed
    )
    return ds


# """
from comptra.utils import is_lang, quality_estimation
from comptra.languages import MAPPING_LANG_TO_KEY
from sonar.models.blaser.loader import load_blaser_model
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
from fairseq2.typing import Device

print(f"device: {device}")
device = Device(device)
blaser_qe = load_blaser_model("blaser_2_0_qe").eval()
blaser_qe.to(device)
text_embedder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=device,
)


def get_blaser_score(x, y, src, tgt):
    src_embs = text_embedder.predict([x], source_lang=MAPPING_LANG_TO_KEY[src])
    ref_embs = text_embedder.predict([y], source_lang=MAPPING_LANG_TO_KEY[tgt])
    blaser_score = blaser_qe(src=src_embs, mt=ref_embs).item()
    return blaser_score


# """
from datasets import load_dataset


def get_all(
    data_dir: str,
    languages: List[str],
    size: int,
    test_size_ratio: Union[float, int],
    seed: int,
    reverse: bool = False,
):
    rng = np.random.default_rng(seed)
    list_of_datasets = []
    for language in languages:
        # """
        list_of_sentences = []
        list_of_candidates = []
        filenames = [
            f"{language}_paraphrase_tear.jsonl",
            f"{language}_paraphrase_maps.jsonl",
            f"{language}_paraphrase_sbys.jsonl",
            f"{language}_paraphrase_refine.jsonl",
            # f"{language}_paraphrase_nllb.jsonl",
            # f"{language}_paraphrase_comptra.jsonl", # Not included in ALL, to reduce the computational cost
        ]
        filenames = [
            filename
            for filename in filenames
            if os.path.exists(os.path.join(data_dir, filename))
        ]
        print(f"filenames: {filenames}")
        list_of_dicos = [
            [json.loads(line) for line in open(os.path.join(data_dir, filename), "r")]
            for filename in filenames
        ]
        assert all([len(element) == len(list_of_dicos[0]) for element in list_of_dicos])
        for j in range(len(list_of_dicos[0])):
            candidates = []
            dico = list_of_dicos[0][j]
            sentence = dico["sentence"]
            translation = dico["translation"]
            if len(sentence.strip()) < 10 or len(translation.strip()) < 10:
                continue
            if any([col in translation for col in ["#", ">"]]) or any(
                [col in sentence for col in ["#", ">"]]
            ):
                continue
            list_of_sentences.append(sentence)
            candidates.append(translation)
            for k in range(len(list_of_dicos)):
                dico = list_of_dicos[k][j]
                if "tear" in filenames[k]:
                    candidates.extend([dico["draft"], dico["refinement"]])
                elif "maps" in filenames[k]:
                    candidates.extend(
                        [
                            dico["zero-shot"],
                            dico["demos-trans"],
                            dico["keywords-trans"],
                            dico["topics-trans"],
                        ]
                    )
                elif "sbys" in filenames[k]:
                    candidates.extend(
                        [dico["draft"], dico["refinement"], dico["proofreading"]]
                    )
                elif "refine" in filenames[k]:
                    candidates.extend(dico["refined_outputs"])
                elif "comptra" in filenames[k]:
                    candidates.extend(dico["translations"])
                elif "nllb" in filenames[k]:
                    candidates.extend([dico["zs_outputs"]])
                else:
                    pass
            list_of_candidates.append(candidates)
        if size < 0:
            selected_indices = [i for i in range(len(list_of_sentences))]
        else:
            selected_indices = rng.choice(
                a=len(list_of_sentences), size=size, replace=False
            ).tolist()

        selected_indices = [i for i in selected_indices if i < len(list_of_sentences)]
        dataset = Dataset.from_dict(
            {
                "source": [list_of_sentences[i] for i in selected_indices],
                "target": [list_of_candidates[i] for i in selected_indices],
                "source_language": ["English"] * len(selected_indices),
                "target_language": [language] * len(selected_indices),
            }
        )
        # """
        # dataset = load_dataset("json", data_files={"train": os.path.join(data_dir, f"{language}_paraphrase_all.json")}, split="train")
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)

    # """
    def f(example):
        translations = []
        for sentence, candidates, src, tgt in zip(
            example["source"],
            example["target"],
            example["source_language"],
            example["target_language"],
        ):
            scores = [
                is_lang(candidate, tgt)
                * get_blaser_score(x=sentence, y=candidate, src=src, tgt=tgt)
                for candidate in candidates
            ]
            translation = candidates[np.argmax(scores)]
            translations.append(translation)
        return {"new_target": translations}

    dataset = dataset.map(lambda x: f(x), batched=True)
    dataset = dataset.remove_columns(["target"])
    dataset = dataset.rename_column("new_target", "target")
    dataset.to_json(os.path.join(data_dir, f"{language}_paraphrase_all.json"))
    # """
    if reverse:
        reverse_dataset = Dataset.from_dict(
            {
                "source": dataset["target"],
                "target": dataset["source"],
                "source_language": dataset["target_language"],
                "target_language": dataset["source_language"],
            }
        )

        print(
            f"We will consider both translation direction: English to {language} and {language} to English."
        )

        dataset = concatenate_datasets([dataset, reverse_dataset]).shuffle(seed=seed)

    if test_size_ratio == 0:
        return dataset

    ds = dataset.train_test_split(
        test_size=(1 + reverse) * test_size_ratio, shuffle=True, seed=seed
    )
    return ds


if __name__ == "__main__":
    pass
