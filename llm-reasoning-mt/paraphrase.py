import argparse
from datasets import load_dataset
from comptra.languages import MAPPING_LANG_TO_KEY
from comptra.prompts.templates import get_template
from comptra.sampler import *
from comptra.retriever import Retriever

from tqdm import tqdm
import numpy as np
import time
import json
import os
import re

from comptra.utils import _stop_at_stop_token, is_lang


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name or path of the model used for text generation.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Name or path of the tokenizer of the model used for text generation",
    )
    parser.add_argument("--api_key", type=str, help="OPENAI API KEY.")
    parser.add_argument(
        "--inference_api",
        type=str,
        default="vllm",
        help="Which API to use for text generation, set to vllm by default.",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=4,
        help="Batch size for the generations.",
    )
    parser.add_argument(
        "--seed", type=int, default=122, help="Seed for random number generation."
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="Maximum number of tokens to generate per query.",
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature of the generation."
    )
    parser.add_argument("--top_p", type=float, help="Nucleus sampling parameter.")
    parser.add_argument("--repetition_penalty", type=float, help="Repetition penalty.")
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        help="Number of responses to return per query.",
    )
    parser.add_argument(
        "--num_beams", type=int, help="Number of beams for beams search."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling in the generation.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to write on the console."
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Languages to generate. A space-separated list of capitalized language names.",
    )
    parser.add_argument(
        "--number_of_generations_per_step",
        type=int,
        help="Number of new text to generate per generation step.",
    )
    parser.add_argument(
        "--source_language",
        type=str,
        default="English",
        help="The target language for back-translation.",
    )
    parser.add_argument("--input_dir", type=str, help="Path to the input directory")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument(
        "--input_filenames",
        nargs="+",
        type=str,
        help="Output filenames for each language we want to generate.",
    )
    parser.add_argument(
        "--template_key",
        type=int,
        help="Name of the template we use for ICL.",
    )
    parser.add_argument(
        "--number_of_demonstrations",
        type=int,
        default=5,
        help="Number of in-context demonstrations for translation.",
    )
    parser.add_argument(
        "--max_samples", type=int, help="Max number of samples (for debugging)."
    )
    parser.add_argument(
        "--cot_template",
        type=int,
        help="Which template to use for the chain of thought.",
    )
    parser.add_argument(
        "--mode", type=str, help="which decomposition template, sp, hard, comptra, p."
    )
    parser.add_argument(
        "--strategy", type=str, help="Which strategy to use.", default="maps"
    )
    return parser.parse_args()


PARAPHRASE = """
We would like to propose a list of paraphrases of sentences. For a given sentence, you will provide {number} paraphrases that have the same meaning as the original sentence and mostly use the same words as well.
Ensure that each of the {number} paraphrases is a correct sentence and does not change the meaning of the original sentence.

Here are some examples with 4 paraphrases per sentence.

<Examples>
Sentence
The Boolean satisfiability problem is a well-researched problem with many exemplar solvers available; it is very fast, as package solving complexity is very low compared to other areas where SAT solvers are used. 

Propositions
    1. The Boolean satisfiability problem is a widely studied topic, with numerous exemplar solvers available; it is efficient, as solving package complexity is significantly lower than in other domains using SAT solvers.
    2. Boolean satisfiability, a well-researched problem, boasts many exemplar solvers, and its speed is notable due to the low complexity of package solving compared to other SAT applications.
    3. The problem of Boolean satisfiability has been extensively researched, leading to the development of many exemplar solvers; package solving in this context is fast, given its comparatively low complexity in contrast to other SAT solver uses.
    4. With numerous exemplar solvers available, the Boolean satisfiability problem is well-researched and demonstrates remarkable speed, as the complexity of package solving is much lower than in other SAT solver applications.

###

Sentence
Dore was offered several one-off shows in night clubs, and her best album was rereleased in 2001. 

Propositions
    1. Dore’s best album was rereleased in 2001, and she was offered several one-off shows in night clubs.
    2. In 2001, Dore’s best album was rereleased, and she received offers for several one-off performances in night clubs.
    3. Several one-off shows in night clubs were offered to Dore, and her best album saw a rerelease in 2001.
    4. Dore was given opportunities for one-off performances in night clubs, and her best album was rereleased during 2001.

###

Sentence
Jim briefly transfers to the Stamford branch after Pam confirmed her commitment to Roy, before corporate is forced to merge the Stamford branch and staff into the Scranton branch.

Propositions
    1. After Pam confirmed her commitment to Roy, Jim briefly transfers to the Stamford branch, only for corporate to merge Stamford staff into the Scranton branch.
    2. Jim transfers briefly to the Stamford branch after Pam confirms her commitment to Roy, but corporate later merges the Stamford staff into the Scranton branch.
    3. Pam's confirmation of her commitment to Roy leads Jim to briefly transfer to the Stamford branch, which is later merged into the Scranton branch by corporate.
    4. Before corporate merges the Stamford branch and its staff into the Scranton branch, Jim briefly transfers there after Pam confirms her commitment to Roy.

###

Sentence
But Jack could not get back to his own time, because one of the drug vials had broken, and there was only enough left in one of the vials to stop Whistler.

Propositions
    1. Jack could not return to his own time because one of the drug vials had broken, leaving only enough in one vial to stop Whistler.
    2. Since one of the drug vials had broken, Jack was unable to get back to his own time, with just enough remaining in a single vial to stop Whistler.
    3. Because one of the vials of the drug had broken, Jack could not make it back to his own time, as only one vial had enough left to stop Whistler.
    4. One of the drug vials had broken, leaving Jack unable to return to his own time, with only enough left in one vial to stop Whistler.
</Examples>

Now, it's your turn. Provide {number} paraphrases that have the same meaning as the following sentence.

Sentence
"""

PARAPHRASE_2 = """
We would like to propose a list of syntactically similar sentences. For a given sentence, provide {number} others with similar syntax. The meaning can differ—what matters is the sentence structure, the grammatical roles (e.g., POS, NER), and the syntactic dependencies (e.g., objects, subjects, modifiers).
Ensure that each of the {number} sentences is correct.

Here are some examples with 4 syntactically similar sentences per sentence.

<Examples>
Sentence
The Boolean satisfiability problem is a well-researched problem with many exemplar solvers available; it is very fast, as package solving complexity is very low compared to other areas where SAT solvers are used. 

Propositions
    1. The protein folding task is a heavily studied topic with several benchmark datasets available; it is quite efficient, as alignment computation cost is minimal compared to other domains where folding tools are applied.
    2. The neural decoding task is a well-known challenge with numerous effective baselines proposed; it performs remarkably well, as classification difficulty is considerably lower than in tasks where deep models are deployed.
    3. The climate prediction problem is a widely explored problem with robust simulation engines developed; it operates rapidly, as parameter estimation difficulty is much lower compared to other problems where such models are employed.
    4. The signal processing pipeline is a highly optimized framework with many reusable modules implemented; it runs quickly, as data transformation overhead is very small compared to other systems where processing chains are used.

###

Sentence
Dore was offered several one-off shows in night clubs, and her best album was rereleased in 2001. 

Propositions
    1. Linda was invited to several solo exhibitions in urban galleries, and her first collection was reprinted in 2010.
    2. Carla was awarded multiple limited-time contracts in high-end resorts, and her breakthrough project was relaunched in 2015.
    3. James was granted various guest appearances on radio shows, and his debut single was remastered in 1998.
    4. Marcus was selected for a few stand-alone gigs in local theaters, and his most popular film was rescreened in 2003.

###

Sentence
Jim briefly transfers to the Stamford branch after Pam confirmed her commitment to Roy, before corporate is forced to merge the Stamford branch and staff into the Scranton branch.

Propositions
    1. Sarah temporarily relocates to the Berlin office after Alex announced her engagement to David, before management decides to consolidate the Berlin and Munich teams.
    2. Lena briefly moves to the Paris division after Nina revealed her loyalty to Ivan, before headquarters begins to integrate the Paris office into the Lyon branch.
    3. Kevin momentarily switches to the New York unit after Laura accepted her proposal with John, before HR initiates the merger of New York operations into the Boston office.
    4. Emily shortly joins the Tokyo department after Rachel disclosed her plans with Hiroshi, before executives order the integration of Tokyo and Kyoto personnel into one division.

###

Sentence
But Jack could not get back to his own time, because one of the drug vials had broken, and there was only enough left in one of the vials to stop Whistler.

Propositions
    1. But Alice couldn't return to her original timeline, because one of the energy cells had leaked, and there was only enough charge left in one cell to power the portal.
    2. But Tom failed to travel back to his former world, because one of the crystals had shattered, and there was only a fragment remaining to stabilize the gate.
    3. But Sarah did not manage to reach her intended reality, because one of the devices had malfunctioned, and there was only enough left in one unit to disrupt the barrier.
    4. But Michael was unable to make it to his future self, because one of the batteries had drained, and there was only sufficient energy left in one pack to stop Magnus.
</Examples>

Now, it's your turn. Provide {number} syntactically similar sentences to the following sentence.

Sentence
"""


def get_prompt_main(sentence, number, mode):
    if mode is None or mode == "p":
        return PARAPHRASE.format(number=number).strip() + f"\n{sentence}"
    else:
        return PARAPHRASE_2.format(number=number).strip() + f"\n{sentence}"


def main(args):
    rng = np.random.default_rng(args.seed)
    languages = args.languages
    print(f"LANGUAGES: {languages}")

    template_key = args.template_key if args.template_key is not None else 11

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": "English",
        "tgt": languages[0],
        "template": get_template(key=template_key, src="English", tgt=languages[0]),
        "merge_prompt": "vanilla",
        "selection_method": "greedy",
        "method_translate": "vanilla",
        "nllb_name_or_path": None,
        "method_divide": None,
    }

    print("FIRST")

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "request_batch_size": args.request_batch_size,
        "verbose": args.verbose,
    }

    if args.inference_api == "vllm":
        sampler = vLLMSampler(**arguments)
    elif args.inference_api == "openai":
        sampler = OpenAISampler(api_key=args.api_key, **arguments)
    elif args.inference_api == "anthropic":
        sampler = AnthropicSampler(**arguments)
    elif args.inference_api == "cohere":
        sampler = cohereSampler(**arguments)
    elif args.inference_api == "hf":
        sampler = HFSampler(**arguments)
    else:
        sampler = Sampler(**arguments)

    input_filenames = (
        args.input_filenames
        if args.input_filenames is not None
        else [f"{language}.jsonl" for language in languages]
    )

    assert len(input_filenames) == len(
        languages
    ), f"The number of input filenames ({len(input_filenames)}) should match the number of languages ({len(languages)})."
    dico_of_inputs = {language: [] for language in languages}
    dico_of_translations = {language: [] for language in languages}

    for i, input_filename in enumerate(input_filenames):
        if os.path.exists(os.path.join(args.input_dir, input_filename)):
            with open(os.path.join(args.input_dir, input_filename), "r") as fin:
                for j, line in enumerate(fin):
                    if args.max_samples is not None and j >= args.max_samples:
                        break
                    """
                    dico_of_inputs[languages[i]].extend(
                        json.loads(line)["translations"]
                    )
                    dico_of_translations[languages[i]].extend(
                        json.loads(line)["propositions"]
                    )
                    """
                    dico_of_inputs[languages[i]].extend(
                        [json.loads(line)["translation"]]
                    )
                    dico_of_translations[languages[i]].extend(
                        [json.loads(line)["sentence"]]
                    )
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            args.input_dir, args.model_name_or_path.split("/")[-1]
        )
    os.makedirs(output_dir, exist_ok=True)

    for i, language in enumerate(languages):
        output_filename = os.path.join(
            output_dir, f"{language}_paraphrase_{args.mode}.jsonl"
        )
        start = 0
        if os.path.exists(output_filename):
            with open(output_filename, "r") as fin:
                for _ in fin:
                    start += 1

        retriever = Retriever(
            source_language=args.source_language,
            dataset_name_or_path="flores",
            retriever_type="bm25s",
            target_language=languages[i],
        )

        sampler.update_template(
            get_template(key=template_key, src=args.source_language, tgt=languages[i])
        )
        sampler.update_src(args.source_language)
        sampler.update_tgt(languages[i])
        for j in range(start, len(dico_of_inputs[language]), args.request_batch_size):
            batch_of_inputs = dico_of_inputs[language][j : j + args.request_batch_size]
            batch_of_translations = dico_of_translations[language][
                j : j + args.request_batch_size
            ]

            prompts = [
                get_prompt_main(
                    sentence, args.number_of_generations_per_step, args.mode
                )
                for sentence in batch_of_inputs
            ]

            print(sampler.apply_chat_template(prompts[0]))

            answers = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in prompts],
                **generation_kwargs,
            )
            list_of_candidates = []
            for r, answer in enumerate(answers):
                answer = answer[0]
                for trigger in [
                    "\n**Propositions:**",
                    "\nPropositions:",
                    "\nPropositions",
                ]:
                    if trigger in answer:
                        answer = answer[answer.find(trigger) + len(trigger) :].strip()
                        answer = answer.split("\n\n")[0]
                answer = _stop_at_stop_token(
                    answer,
                    STOP_WORDS + [f"{args.number_of_generations_per_step + 1}.\t"],
                )
                if "1. \n" in answer:
                    pattern = r"(\d+\. \n)"
                elif "1. " in answer:
                    pattern = r"(\d+\. )"
                    # pattern = r"(\d+\. ?(?:\n)?)"
                else:
                    pattern = r"(\d+\.\n)"
                splitted_answer = re.split(pattern, answer)
                try:
                    assert (
                        len(splitted_answer)
                        == 2 * args.number_of_generations_per_step + 1
                    ), "There is an issue."
                except Exception as exc:
                    print(
                        f"The generation {r + 1} does not have the required quality. Pattern = {pattern}, length = {len(splitted_answer)}"
                    )
                    splitted_answer = []
                candidates = []
                for j, element in enumerate(splitted_answer):
                    if j == 0:
                        # everything before the first match
                        continue
                    if j % 2 == 1:
                        # matches the iterator
                        continue
                    candidate = element.strip()
                    candidate = [
                        element.strip()
                        for element in candidate.split("\n")
                        if element.strip() != ""
                    ]
                    if len(candidate) == 0:
                        continue
                    candidate = candidate[0]
                    candidates.append(candidate)

                list_of_candidates.append(candidates)
            # Flatten the list of sentences
            flat_list_of_candidates = [
                candidate
                for candidates in list_of_candidates
                for candidate in candidates
            ]
            if args.number_of_demonstrations > 0:
                batch_of_demonstrations = [
                    retriever.query(sentence=sentence, k=args.number_of_demonstrations)
                    for sentence in flat_list_of_candidates
                ]
            else:
                batch_of_demonstrations = [[] for _ in flat_list_of_candidates]

            outputs = sampler.translate(
                sentences=flat_list_of_candidates,
                demonstrations=batch_of_demonstrations,
                **generation_kwargs,
            )

            current_index = 0
            for p in range(len(list_of_candidates)):
                sentences = list_of_candidates[p]
                translated_sentences = outputs[
                    current_index : current_index + len(sentences)
                ]
                # Filter the sentences
                correct_language_indices = [
                    q
                    for q in range(len(translated_sentences))
                    if is_lang(translated_sentences[q], language)  # Right language
                    # and len(translated_sentences[q]) >= 40  # Long enough
                    and translated_sentences[q].strip().endswith(".")  # Ends with a dot
                ]
                # save the translations
                if args.verbose:
                    for k in range(len(sentences)):
                        print(
                            f"{current_index + k + 1}. T -> {translated_sentences[k]}\nS -> {sentences[k]}"
                        )
                    print(
                        f"{len(correct_language_indices)}/{len(sentences)} are in the correct language and have the right format."
                    )
                with open(
                    os.path.join(output_dir, output_filename), "a", encoding="utf-8"
                ) as fout:
                    dico = {
                        "sentence": batch_of_inputs[p],
                        "translation": batch_of_translations[p],
                        "paraphrases": [sentences[q] for q in correct_language_indices],
                        "translations": [
                            translated_sentences[q] for q in correct_language_indices
                        ],
                    }
                    fout.write(json.dumps(dico) + "\n")
                current_index += len(sentences)
    print("END")


PARAPHRASE_SECOND = """
We would like you to propose a list of paraphrases of sentences. For a given sentence, you will provide {number} paraphrases that have the same meaning as the original sentence and mostly use the same words as well.
Ensure that each of the {number} paraphrases is a correct sentence and does not change the meaning of the original sentence.

Here are some examples for English sentences with 4 paraphrases per sentence.

<Examples>
Sentence
The Boolean satisfiability problem is a well-researched problem with many exemplar solvers available; it is very fast, as package solving complexity is very low compared to other areas where SAT solvers are used. 

Propositions
    1. The Boolean satisfiability problem is a widely studied topic, with numerous exemplar solvers available; it is efficient, as solving package complexity is significantly lower than in other domains using SAT solvers.
    2. Boolean satisfiability, a well-researched problem, boasts many exemplar solvers, and its speed is notable due to the low complexity of package solving compared to other SAT applications.
    3. The problem of Boolean satisfiability has been extensively researched, leading to the development of many exemplar solvers; package solving in this context is fast, given its comparatively low complexity in contrast to other SAT solver uses.
    4. With numerous exemplar solvers available, the Boolean satisfiability problem is well-researched and demonstrates remarkable speed, as the complexity of package solving is much lower than in other SAT solver applications.

###

Sentence
Dore was offered several one-off shows in night clubs, and her best album was rereleased in 2001. 

Propositions
    1. Dore’s best album was rereleased in 2001, and she was offered several one-off shows in night clubs.
    2. In 2001, Dore’s best album was rereleased, and she received offers for several one-off performances in night clubs.
    3. Several one-off shows in night clubs were offered to Dore, and her best album saw a rerelease in 2001.
    4. Dore was given opportunities for one-off performances in night clubs, and her best album was rereleased during 2001.

###

Sentence
Jim briefly transfers to the Stamford branch after Pam confirmed her commitment to Roy, before corporate is forced to merge the Stamford branch and staff into the Scranton branch.

Propositions
    1. After Pam confirmed her commitment to Roy, Jim briefly transfers to the Stamford branch, only for corporate to merge Stamford staff into the Scranton branch.
    2. Jim transfers briefly to the Stamford branch after Pam confirms her commitment to Roy, but corporate later merges the Stamford staff into the Scranton branch.
    3. Pam's confirmation of her commitment to Roy leads Jim to briefly transfer to the Stamford branch, which is later merged into the Scranton branch by corporate.
    4. Before corporate merges the Stamford branch and its staff into the Scranton branch, Jim briefly transfers there after Pam confirms her commitment to Roy.

###

Sentence
But Jack could not get back to his own time, because one of the drug vials had broken, and there was only enough left in one of the vials to stop Whistler.

Propositions
    1. Jack could not return to his own time because one of the drug vials had broken, leaving only enough in one vial to stop Whistler.
    2. Since one of the drug vials had broken, Jack was unable to get back to his own time, with just enough remaining in a single vial to stop Whistler.
    3. Because one of the vials of the drug had broken, Jack could not make it back to his own time, as only one vial had enough left to stop Whistler.
    4. One of the drug vials had broken, leaving Jack unable to return to his own time, with only enough left in one vial to stop Whistler.
</Examples>

Now, it's your turn. Provide {number} paraphrases that have the same meaning as the following {lang} sentence. Do not translate the sentence, just write the paraphrases directly in the correct language (in this case, {lang}).
Only write the paraphrases as a numbered list, do not correct them afterwards.

Sentence
"""


def get_prompt_2(sentence, number, lang):
    return (
        PARAPHRASE_SECOND.format(number=number, lang=lang).strip()
        + f"\nSentence\n{sentence}"
    )


def second(args):
    rng = np.random.default_rng(args.seed)
    languages = args.languages
    print(f"LANGUAGES: {languages}")

    template_key = args.template_key if args.template_key is not None else 11

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": languages[0],
        "tgt": args.source_language,
        "template": get_template(key=template_key, src=languages[0], tgt="English"),
        "merge_prompt": "vanilla",
        "selection_method": "greedy",
        "method_translate": "vanilla",
        "nllb_name_or_path": None,
        "method_divide": None,
    }

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "request_batch_size": args.request_batch_size,
        "verbose": args.verbose,
    }

    if args.inference_api == "vllm":
        sampler = vLLMSampler(**arguments)
    elif args.inference_api == "openai":
        sampler = OpenAISampler(api_key=args.api_key, **arguments)
    elif args.inference_api == "anthropic":
        sampler = AnthropicSampler(**arguments)
    elif args.inference_api == "cohere":
        sampler = cohereSampler(**arguments)
    elif args.inference_api == "hf":
        sampler = HFSampler(**arguments)
    else:
        sampler = Sampler(**arguments)

    input_filenames = (
        args.input_filenames
        if args.input_filenames is not None
        else [f"{language}.jsonl" for language in languages]
    )

    assert len(input_filenames) == len(
        languages
    ), f"The number of input filenames ({len(input_filenames)}) should match the number of languages ({len(languages)})."
    dico_of_inputs = {language: [] for language in languages}
    dico_of_translations = {language: [] for language in languages}

    for i, input_filename in enumerate(input_filenames):
        if os.path.exists(os.path.join(args.input_dir, input_filename)):
            with open(os.path.join(args.input_dir, input_filename), "r") as fin:
                for j, line in enumerate(fin):
                    if args.max_samples is not None and j >= args.max_samples:
                        break
                    """
                    dico_of_inputs[languages[i]].extend(
                        json.loads(line)["translations"]
                    )
                    dico_of_translations[languages[i]].extend(
                        json.loads(line)["propositions"]
                    )
                    """
                    dico_of_inputs[languages[i]].extend(
                        [json.loads(line)["translation"]]
                    )
                    dico_of_translations[languages[i]].extend(
                        [json.loads(line)["sentence"]]
                    )
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            args.input_dir, args.model_name_or_path.split("/")[-1]
        )
    os.makedirs(output_dir, exist_ok=True)

    for i, language in enumerate(languages):
        output_filename = os.path.join(
            output_dir, f"{language}_paraphrase_{args.mode}.jsonl"
        )
        start = 0
        if os.path.exists(output_filename):
            with open(output_filename, "r") as fin:
                for _ in fin:
                    start += 1

        retriever = Retriever(
            source_language=languages[i],
            dataset_name_or_path="flores",
            retriever_type="bm25s",
            target_language=args.source_language,
        )

        sampler.update_template(
            get_template(key=template_key, src=languages[i], tgt=args.source_language)
        )
        sampler.update_src(languages[i])
        sampler.update_tgt(args.source_language)
        for j in range(start, len(dico_of_inputs[language]), args.request_batch_size):
            batch_of_inputs = dico_of_inputs[language][j : j + args.request_batch_size]
            batch_of_translations = dico_of_translations[language][
                j : j + args.request_batch_size
            ]

            prompts = [
                get_prompt_2(sentence, args.number_of_generations_per_step, language)
                for sentence in batch_of_translations
            ]
            answers = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in prompts],
                **generation_kwargs,
            )
            list_of_candidates = []
            for r, answer in enumerate(answers):
                answer = answer[0]
                answer = _stop_at_stop_token(
                    answer,
                    [f"{args.number_of_generations_per_step + 1}.\t"] + STOP_WORDS,
                )
                if "1. \n" in answer:
                    pattern = r"(\d+\. \n)"
                elif "1. " in answer:
                    # pattern = r"(\d+\. )"
                    pattern = r"(\d+\. ?(?:\n)?)"
                else:
                    pattern = r"(\d+\.\n)"
                splitted_answer = re.split(pattern, answer)
                try:
                    assert (
                        len(splitted_answer)
                        == 2 * args.number_of_generations_per_step + 1
                    ), "There is an issue."
                except Exception as exc:
                    print(
                        f"The generation does not have the required quality. Pattern = {pattern}, length = {len(splitted_answer)}"
                    )
                    print(f"===\nSTART\n===\n{answer}\n===\nEND\n===")
                    splitted_answer = []
                candidates = []
                for j, element in enumerate(splitted_answer):
                    if j == 0:
                        # everything before the first match
                        continue
                    if j % 2 == 1:
                        # matches the iterator
                        continue
                    candidate = element.strip()
                    candidate = [
                        element.strip()
                        for element in candidate.split("\n")
                        if element.strip() != ""
                    ]
                    if len(candidate) == 0:
                        continue
                    candidate = candidate[0]
                    candidates.append(candidate)

                list_of_candidates.append(candidates)
            # Flatten the list of sentences
            flat_list_of_candidates = [
                candidate
                for candidates in list_of_candidates
                for candidate in candidates
            ]
            if args.number_of_demonstrations > 0:
                batch_of_demonstrations = [
                    retriever.query(sentence=sentence, k=args.number_of_demonstrations)
                    for sentence in flat_list_of_candidates
                ]
            else:
                batch_of_demonstrations = [[] for _ in flat_list_of_candidates]

            if len(flat_list_of_candidates) > 0:
                outputs = sampler.translate(
                    sentences=flat_list_of_candidates,
                    demonstrations=batch_of_demonstrations,
                    **generation_kwargs,
                )
            else:
                outputs = []

            current_index = 0
            for p in range(len(list_of_candidates)):
                sentences = list_of_candidates[p]
                translated_sentences = outputs[
                    current_index : current_index + len(sentences)
                ]
                # Filter the sentences
                correct_language_indices = [
                    q
                    for q in range(len(translated_sentences))
                    if is_lang(
                        translated_sentences[q], args.source_language
                    )  # Right language
                    and len(translated_sentences[q]) >= 40  # Long enough
                    and translated_sentences[q].strip().endswith(".")  # Ends with a dot
                ]
                # save the translations
                if args.verbose:
                    for k in range(len(sentences)):
                        print(
                            f"{current_index + k + 1}. T -> {translated_sentences[k]}\nS -> {sentences[k]}"
                        )
                    print(
                        f"{len(correct_language_indices)}/{len(sentences)} are in the correct language and have the right format."
                    )
                with open(
                    os.path.join(output_dir, output_filename), "a", encoding="utf-8"
                ) as fout:
                    dico = {
                        "sentence": batch_of_inputs[p],
                        "translation": batch_of_translations[p],
                        "paraphrases": [
                            translated_sentences[q] for q in correct_language_indices
                        ],
                        "translations": [
                            sentences[q] for q in correct_language_indices
                        ],
                    }
                    fout.write(json.dumps(dico) + "\n")
                current_index += len(sentences)
    print("END")


def third(args):
    rng = np.random.default_rng(args.seed)
    languages = args.languages
    print(f"LANGUAGES: {languages}")

    template_key = args.template_key if args.template_key is not None else 11

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": languages[0],
        "tgt": args.source_language,
        "template": get_template(key=template_key, src=languages[0], tgt="English"),
        "merge_prompt": "vanilla",
        "selection_method": "greedy",
        "method_translate": "vanilla",
        "nllb_name_or_path": None,
        "method_divide": None,
    }

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "request_batch_size": args.request_batch_size,
        "verbose": args.verbose,
    }

    if args.inference_api == "vllm":
        sampler = vLLMSampler(**arguments)
    elif args.inference_api == "openai":
        sampler = OpenAISampler(api_key=args.api_key, **arguments)
    elif args.inference_api == "anthropic":
        sampler = AnthropicSampler(**arguments)
    elif args.inference_api == "cohere":
        sampler = cohereSampler(**arguments)
    elif args.inference_api == "hf":
        sampler = HFSampler(**arguments)
    else:
        sampler = Sampler(**arguments)

    input_filenames = (
        args.input_filenames
        if args.input_filenames is not None
        else [f"{language}.jsonl" for language in languages]
    )

    assert len(input_filenames) == len(
        languages
    ), f"The number of input filenames ({len(input_filenames)}) should match the number of languages ({len(languages)})."
    dico_of_inputs = {language: [] for language in languages}
    dico_of_translations = {language: [] for language in languages}

    for i, input_filename in enumerate(input_filenames):
        if os.path.exists(os.path.join(args.input_dir, input_filename)):
            with open(os.path.join(args.input_dir, input_filename), "r") as fin:
                for j, line in enumerate(fin):
                    if args.max_samples is not None and j >= args.max_samples:
                        break
                    """
                    dico_of_inputs[languages[i]].extend(
                        json.loads(line)["translations"]
                    )
                    dico_of_translations[languages[i]].extend(
                        json.loads(line)["propositions"]
                    )
                    """
                    dico_of_inputs[languages[i]].extend(
                        [json.loads(line)["translation"]]
                    )
                    dico_of_translations[languages[i]].extend(
                        [json.loads(line)["sentence"]]
                    )
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            args.input_dir, args.model_name_or_path.split("/")[-1]
        )
    os.makedirs(output_dir, exist_ok=True)

    for i, language in enumerate(languages):
        output_filename = os.path.join(output_dir, f"{language}_paraphrase_sbys.jsonl")
        start = 0
        if os.path.exists(output_filename):
            with open(output_filename, "r") as fin:
                for _ in fin:
                    start += 1
        sampler.update_template(
            get_template(key=template_key, src=args.source_language, tgt=languages[i])
        )
        sampler.update_src(args.source_language)
        sampler.update_tgt(languages[i])

        for j in range(start, len(dico_of_inputs[language]), args.request_batch_size):
            batch_of_inputs = dico_of_inputs[language][j : j + args.request_batch_size]
            batch_of_translations = dico_of_translations[language][
                j : j + args.request_batch_size
            ]
            # Pre-drafting research
            pre_translation_prompts = [
                get_step_by_step_prompts(
                    description="pre-translation-research",
                    src=args.source_language,
                    tgt=languages[i],
                    source=sentence,
                )
                for sentence in batch_of_inputs
            ]

            pre_translation_outputs = sampler.generate(
                [
                    sampler.apply_chat_template(prompt)
                    for prompt in pre_translation_prompts
                ],
                **generation_kwargs,
            )
            # Drafting
            draft_prompts = [
                [
                    {"role": "user", "content": pre_translation_prompts[k]},
                    {
                        "role": "assistant",
                        "content": pre_translation_outputs[k][0].strip(),
                    },
                    {
                        "role": "user",
                        "content": get_step_by_step_prompts(
                            description="drafting",
                            src=args.source_language,
                            source=batch_of_inputs[k],
                        ),
                    },
                ]
                for k in range(len(batch_of_inputs))
            ]

            draft_outputs = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in draft_prompts],
                **generation_kwargs,
            )
            # Refinement
            refine_prompts = [
                draft_prompts[k]
                + [
                    {"role": "assistant", "content": draft_outputs[k][0].strip()},
                    {
                        "role": "user",
                        "content": get_step_by_step_prompts(description="refinement"),
                    },
                ]
                for k in range(len(batch_of_inputs))
            ]

            refine_outputs = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in refine_prompts],
                **generation_kwargs,
            )
            # Proof reading, new conversation
            proofreading_prompts = [
                get_step_by_step_prompts(
                    description="proofreading",
                    source=batch_of_inputs[k],
                    draft=draft_outputs[k][0].strip(),
                    refine=refine_outputs[k][0].strip().split("\n")[0].strip(),
                )
                for k in range(len(batch_of_inputs))
            ]

            proofreading_outputs = sampler.generate(
                [
                    sampler.apply_chat_template(prompt)
                    for prompt in proofreading_prompts
                ],
                **generation_kwargs,
            )

            with open(
                os.path.join(output_dir, output_filename), "a", encoding="utf-8"
            ) as fout:
                for k in range(len(batch_of_inputs)):
                    fout.write(
                        json.dumps(
                            {
                                "sentence": batch_of_inputs[k],
                                "translation": batch_of_translations[k],
                                "research": pre_translation_outputs[k][0].strip(),
                                "draft": draft_outputs[k][0].strip(),
                                "refinement": refine_outputs[k][0]
                                .strip()
                                .split("\n")[0]
                                .strip(),
                                "proofreading": proofreading_outputs[k][0]
                                .strip()
                                .split("\n")[0]
                                .strip(),
                            }
                        )
                        + "\n"
                    )
    print("END")


def fourth(args):
    rng = np.random.default_rng(args.seed)
    languages = args.languages
    print(f"LANGUAGES: {languages}")

    template_key = args.template_key if args.template_key is not None else 11

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": languages[0],
        "tgt": args.source_language,
        "template": get_template(key=template_key, src=languages[0], tgt="English"),
        "merge_prompt": "vanilla",
        "selection_method": "greedy",
        "method_translate": "vanilla",
        "nllb_name_or_path": None,
        "method_divide": None,
    }

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "request_batch_size": args.request_batch_size,
        "verbose": args.verbose,
    }

    if args.inference_api == "vllm":
        sampler = vLLMSampler(**arguments)
    elif args.inference_api == "openai":
        sampler = OpenAISampler(api_key=args.api_key, **arguments)
    elif args.inference_api == "anthropic":
        sampler = AnthropicSampler(**arguments)
    elif args.inference_api == "cohere":
        sampler = cohereSampler(**arguments)
    elif args.inference_api == "hf":
        sampler = HFSampler(**arguments)
    else:
        sampler = Sampler(**arguments)

    input_filenames = (
        args.input_filenames
        if args.input_filenames is not None
        else [f"{language}.jsonl" for language in languages]
    )

    assert len(input_filenames) == len(
        languages
    ), f"The number of input filenames ({len(input_filenames)}) should match the number of languages ({len(languages)})."
    dico_of_inputs = {language: [] for language in languages}
    dico_of_translations = {language: [] for language in languages}

    for i, input_filename in enumerate(input_filenames):
        if os.path.exists(os.path.join(args.input_dir, input_filename)):
            with open(os.path.join(args.input_dir, input_filename), "r") as fin:
                for j, line in enumerate(fin):
                    if args.max_samples is not None and j >= args.max_samples:
                        break
                    """
                    dico_of_inputs[languages[i]].extend(
                        json.loads(line)["translations"]
                    )
                    dico_of_translations[languages[i]].extend(
                        json.loads(line)["propositions"]
                    )
                    """
                    dico_of_inputs[languages[i]].extend(
                        [json.loads(line)["translation"]]
                    )
                    dico_of_translations[languages[i]].extend(
                        [json.loads(line)["sentence"]]
                    )
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            args.input_dir, args.model_name_or_path.split("/")[-1]
        )
    os.makedirs(output_dir, exist_ok=True)

    for i, language in enumerate(languages):
        output_filename = os.path.join(output_dir, f"{language}_paraphrase_maps.jsonl")
        start = 0
        if os.path.exists(output_filename):
            with open(output_filename, "r") as fin:
                for _ in fin:
                    start += 1
        sampler.update_template(
            get_template(key=template_key, src=args.source_language, tgt=languages[i])
        )
        sampler.update_src(args.source_language)
        sampler.update_tgt(languages[i])

        for j in range(start, len(dico_of_inputs[language]), args.request_batch_size):
            batch_of_inputs = dico_of_inputs[language][j : j + args.request_batch_size]
            batch_of_translations = dico_of_translations[language][
                j : j + args.request_batch_size
            ]
            # Ask for Demonstrations
            demos_prompts = [
                f"Provide only the new {args.source_language}-{languages[i]} pair, nothing else.\n\n"
                + get_maps_aspects(
                    sentence=sentence,
                    src=args.source_language,
                    tgt=languages[i],
                    description="demos",
                )
                for sentence in batch_of_inputs
            ]
            print(sampler.apply_chat_template(demos_prompts[0]))
            demos_outputs = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in demos_prompts],
                **generation_kwargs,
            )
            demos = [element[0].strip().split("\n\n")[0] for element in demos_outputs]
            # Ask for keywords
            keywords_prompts = [
                "Provide only the keywords from the given sentence and their translations, nothing else.\n\n"
                + get_maps_aspects(
                    sentence=sentence,
                    src=args.source_language,
                    tgt=languages[i],
                    description="keywords",
                )
                for sentence in batch_of_inputs
            ]
            print(sampler.apply_chat_template(keywords_prompts[0]))
            keywords_outputs = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in keywords_prompts],
                **generation_kwargs,
            )
            keywords = [
                element[0].strip().split("\n\n")[-1].strip()
                for element in keywords_outputs
            ]
            triggers = [
                "services = iinkonzo\n",
                "AOL\n",
                "Microsoft\n",
                "services = iinkonzo \n",
                "AOL \n",
                "Microsoft \n",
            ]
            trimmed_keywords = []
            for keyword in keywords:
                start_keyword = 0
                for trigger in triggers:
                    if keyword.find(trigger) >= 0:
                        start_keyword = max(
                            start_keyword, keyword.find(trigger) + len(trigger)
                        )
                trimmed_keywords.append(keyword[start_keyword:].strip())
            # Ask for topics
            topics_prompts = [
                "Provide only the topics of the given sentence, nothing else.\n\n"
                + get_maps_aspects(
                    sentence=sentence,
                    src=args.source_language,
                    tgt=languages[i],
                    description="topics",
                )
                for sentence in batch_of_inputs
            ]
            print(sampler.apply_chat_template(topics_prompts[0]))
            topics_outputs = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in topics_prompts],
                **generation_kwargs,
            )
            topics = [element[0].strip().split("\n\n")[0] for element in topics_outputs]
            # Knowledge integration: demonstrations
            demos_trans_prompts = [
                "Please provide only the translation, nothing more.\n\n"
                + get_maps_aspects(
                    sentence=sentence,
                    src=args.source_language,
                    tgt=languages[i],
                    description="trans-demos",
                    demos=demo,
                )
                for sentence, demo in zip(batch_of_inputs, demos)
            ]
            print(sampler.apply_chat_template(demos_trans_prompts[0]))
            demos_trans_outputs = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in demos_trans_prompts],
                **generation_kwargs,
            )
            # Knowledge integration: keywords
            keywords_trans_prompts = [
                "Please provide only the translation, nothing more.\n\n"
                + get_maps_aspects(
                    sentence=sentence,
                    src=args.source_language,
                    tgt=languages[i],
                    description="trans-keywords",
                    keywords=keyword,
                )
                for sentence, keyword in zip(batch_of_inputs, keywords)
            ]
            print(sampler.apply_chat_template(keywords_trans_prompts[0]))
            keywords_trans_outputs = sampler.generate(
                [
                    sampler.apply_chat_template(prompt)
                    for prompt in keywords_trans_prompts
                ],
                **generation_kwargs,
            )
            # Knowledge integration: topics
            topics_trans_prompts = [
                "Please provide only the translation, nothing more.\n\n"
                + get_maps_aspects(
                    sentence=sentence,
                    src=args.source_language,
                    tgt=languages[i],
                    description="trans-topics",
                    topics=topic,
                )
                for sentence, topic in zip(batch_of_inputs, topics)
            ]
            print(sampler.apply_chat_template(topics_trans_prompts[0]))
            topics_trans_outputs = sampler.generate(
                [
                    sampler.apply_chat_template(prompt)
                    for prompt in topics_trans_prompts
                ],
                **generation_kwargs,
            )
            # Zero-shot outputs
            zs_outputs = sampler.translate(
                sentences=batch_of_inputs,
                demonstrations=[[] for _ in range(len(batch_of_inputs))],
                **generation_kwargs,
            )
            with open(
                os.path.join(output_dir, output_filename), "a", encoding="utf-8"
            ) as fout:
                for k in range(len(batch_of_inputs)):
                    fout.write(
                        json.dumps(
                            {
                                "sentence": batch_of_inputs[k],
                                "translation": batch_of_translations[k],
                                "keywords": keywords[k],
                                "topics": topics[k],
                                "demonstrations": demos[k],
                                "demos-trans": demos_trans_outputs[k][0]
                                .strip()
                                .split("\n")[0]
                                .strip(),
                                "keywords-trans": keywords_trans_outputs[k][0]
                                .strip()
                                .split("\n")[0]
                                .strip(),
                                "topics-trans": topics_trans_outputs[k][0]
                                .strip()
                                .split("\n")[0]
                                .strip(),
                                "zero-shot": zs_outputs[k],
                            }
                        )
                        + "\n"
                    )
    print("END")


def fifth(args):
    rng = np.random.default_rng(args.seed)
    languages = args.languages
    print(f"LANGUAGES: {languages}")

    template_key = args.template_key if args.template_key is not None else 11

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": languages[0],
        "tgt": args.source_language,
        "template": get_template(key=template_key, src=languages[0], tgt="English"),
        "merge_prompt": "vanilla",
        "selection_method": "greedy",
        "method_translate": "vanilla",
        "nllb_name_or_path": None,
        "method_divide": None,
    }

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "request_batch_size": args.request_batch_size,
        "verbose": args.verbose,
    }

    if args.inference_api == "vllm":
        sampler = vLLMSampler(**arguments)
    elif args.inference_api == "openai":
        sampler = OpenAISampler(api_key=args.api_key, **arguments)
    elif args.inference_api == "anthropic":
        sampler = AnthropicSampler(**arguments)
    elif args.inference_api == "cohere":
        sampler = cohereSampler(**arguments)
    elif args.inference_api == "hf":
        sampler = HFSampler(**arguments)
    else:
        sampler = Sampler(**arguments)

    input_filenames = (
        args.input_filenames
        if args.input_filenames is not None
        else [f"{language}.jsonl" for language in languages]
    )

    assert len(input_filenames) == len(
        languages
    ), f"The number of input filenames ({len(input_filenames)}) should match the number of languages ({len(languages)})."
    dico_of_inputs = {language: [] for language in languages}
    dico_of_translations = {language: [] for language in languages}

    for i, input_filename in enumerate(input_filenames):
        if os.path.exists(os.path.join(args.input_dir, input_filename)):
            with open(os.path.join(args.input_dir, input_filename), "r") as fin:
                for j, line in enumerate(fin):
                    if args.max_samples is not None and j >= args.max_samples:
                        break
                    """
                    dico_of_inputs[languages[i]].extend(
                        json.loads(line)["translations"]
                    )
                    dico_of_translations[languages[i]].extend(
                        json.loads(line)["propositions"]
                    )
                    """
                    dico_of_inputs[languages[i]].extend(
                        [json.loads(line)["translation"]]
                    )
                    dico_of_translations[languages[i]].extend(
                        [json.loads(line)["sentence"]]
                    )
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            args.input_dir, args.model_name_or_path.split("/")[-1]
        )
    os.makedirs(output_dir, exist_ok=True)

    for i, language in enumerate(languages):
        output_filename = os.path.join(
            output_dir, f"{language}_paraphrase_refine.jsonl"
        )
        start = 0
        if os.path.exists(output_filename):
            with open(output_filename, "r") as fin:
                for _ in fin:
                    start += 1
        sampler.update_template(
            get_template(key=template_key, src=args.source_language, tgt=languages[i])
        )
        sampler.update_src(args.source_language)
        sampler.update_tgt(languages[i])

        for j in range(start, len(dico_of_inputs[language]), args.request_batch_size):
            batch_of_inputs = dico_of_inputs[language][j : j + args.request_batch_size]
            batch_of_translations = dico_of_translations[language][
                j : j + args.request_batch_size
            ]
            # Zero-shot outputs
            zs_outputs = sampler.translate(
                sentences=batch_of_inputs,
                demonstrations=[[] for _ in range(len(batch_of_inputs))],
                **generation_kwargs,
            )
            batch_of_outputs = [[element] for element in zs_outputs]  # List[List[str]]
            prev_translations = zs_outputs
            if args.number_of_generations_per_step is None:
                number_of_generations_per_step = 5
            else:
                number_of_generations_per_step = args.number_of_generations_per_step
            for _ in range(number_of_generations_per_step):
                prompts = [
                    get_refine_prompt(
                        source=sentence,
                        prev_translation=translation,
                        src=args.source_language,
                        tgt=languages[i],
                    )
                    for sentence, translation in zip(batch_of_inputs, prev_translations)
                ]
                outputs = sampler.generate(
                    [sampler.apply_chat_template(prompt) for prompt in prompts],
                    **generation_kwargs,
                )
                next_translations = [
                    output[0].strip().split("\n")[0].strip() for output in outputs
                ]
                for k in range(len(batch_of_inputs)):
                    batch_of_outputs[k].append(next_translations[k])
                prev_translations = next_translations
            batch_of_outputs = [
                [element for element in outputs if is_lang(element, languages[i])]
                for outputs in batch_of_outputs
            ]
            with open(
                os.path.join(output_dir, output_filename), "a", encoding="utf-8"
            ) as fout:
                for k in range(len(batch_of_inputs)):
                    fout.write(
                        json.dumps(
                            {
                                "sentence": batch_of_inputs[k],
                                "translation": batch_of_translations[k],
                                "refined_outputs": batch_of_outputs[k],
                            }
                        )
                        + "\n"
                    )
    print("END")


def sixth(args):
    rng = np.random.default_rng(args.seed)
    languages = args.languages
    print(f"LANGUAGES: {languages}")

    template_key = args.template_key if args.template_key is not None else 11

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": languages[0],
        "tgt": args.source_language,
        "template": get_template(key=template_key, src=languages[0], tgt="English"),
        "merge_prompt": "vanilla",
        "selection_method": "greedy",
        "method_translate": "vanilla",
        "nllb_name_or_path": None,
        "method_divide": None,
    }

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "request_batch_size": args.request_batch_size,
        "verbose": args.verbose,
    }

    if args.inference_api == "vllm":
        sampler = vLLMSampler(**arguments)
    elif args.inference_api == "openai":
        sampler = OpenAISampler(api_key=args.api_key, **arguments)
    elif args.inference_api == "anthropic":
        sampler = AnthropicSampler(**arguments)
    elif args.inference_api == "cohere":
        sampler = cohereSampler(**arguments)
    elif args.inference_api == "hf":
        sampler = HFSampler(**arguments)
    else:
        sampler = Sampler(**arguments)

    input_filenames = (
        args.input_filenames
        if args.input_filenames is not None
        else [f"{language}.jsonl" for language in languages]
    )

    assert len(input_filenames) == len(
        languages
    ), f"The number of input filenames ({len(input_filenames)}) should match the number of languages ({len(languages)})."
    dico_of_inputs = {language: [] for language in languages}
    dico_of_translations = {language: [] for language in languages}

    for i, input_filename in enumerate(input_filenames):
        if os.path.exists(os.path.join(args.input_dir, input_filename)):
            with open(os.path.join(args.input_dir, input_filename), "r") as fin:
                for j, line in enumerate(fin):
                    if args.max_samples is not None and j >= args.max_samples:
                        break
                    """
                    dico_of_inputs[languages[i]].extend(
                        json.loads(line)["translations"]
                    )
                    dico_of_translations[languages[i]].extend(
                        json.loads(line)["propositions"]
                    )
                    """
                    dico_of_inputs[languages[i]].extend(
                        [json.loads(line)["translation"]]
                    )
                    dico_of_translations[languages[i]].extend(
                        [json.loads(line)["sentence"]]
                    )
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            args.input_dir, args.model_name_or_path.split("/")[-1]
        )
    os.makedirs(output_dir, exist_ok=True)

    for i, language in enumerate(languages):
        output_filename = os.path.join(output_dir, f"{language}_paraphrase_tear.jsonl")
        start = 0
        if os.path.exists(output_filename):
            with open(output_filename, "r") as fin:
                for _ in fin:
                    start += 1

        retriever = Retriever(
            source_language=args.source_language,
            dataset_name_or_path="flores",
            retriever_type="bm25s",
            target_language=languages[i],
        )

        sampler.update_template(
            get_template(key=template_key, src=args.source_language, tgt=languages[i])
        )
        sampler.update_src(args.source_language)
        sampler.update_tgt(languages[i])

        for j in range(start, len(dico_of_inputs[language]), args.request_batch_size):
            batch_of_inputs = dico_of_inputs[language][j : j + args.request_batch_size]
            batch_of_translations = dico_of_translations[language][
                j : j + args.request_batch_size
            ]
            batch_of_demonstrations = [
                retriever.query(sentence=sentence, k=args.number_of_demonstrations)
                for sentence in batch_of_inputs
            ]
            # Few-shot outputs
            translation_outputs = sampler.translate(
                sentences=batch_of_inputs,
                demonstrations=batch_of_demonstrations,
                **generation_kwargs,
            )
            # Estimate outputs
            estimate_prompts = [
                get_tear_prompts(
                    description="estimate",
                    src=args.source_language,
                    tgt=languages[i],
                    source=sentence,
                    draft=trans,
                )
                for (sentence, trans) in zip(batch_of_inputs, translation_outputs)
            ]
            print(sampler.apply_chat_template(estimate_prompts[0]))
            estimate_outputs = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in estimate_prompts],
                **generation_kwargs,
            )
            # Refine outputs
            refine_prompts = [
                get_tear_prompts(
                    description="refine",
                    src=args.source_language,
                    tgt=languages[i],
                    source=sentence,
                    draft=trans,
                    demonstrations=demonstrations,
                    estimate_fdb=estimate_output[0].strip(),
                )
                for (sentence, trans, demonstrations, estimate_output) in zip(
                    batch_of_inputs,
                    translation_outputs,
                    batch_of_demonstrations,
                    estimate_outputs,
                )
            ]
            print(sampler.apply_chat_template(refine_prompts[0]))
            refine_outputs = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in refine_prompts],
                **generation_kwargs,
            )
            with open(
                os.path.join(output_dir, output_filename), "a", encoding="utf-8"
            ) as fout:
                for k in range(len(batch_of_inputs)):
                    fout.write(
                        json.dumps(
                            {
                                "sentence": batch_of_inputs[k],
                                "translation": batch_of_translations[k],
                                "draft": translation_outputs[k],
                                "estimation": estimate_outputs[k][0].strip(),
                                "refinement": refine_outputs[k][0]
                                .strip()
                                .split("\n")[0]
                                .strip(),
                            }
                        )
                        + "\n"
                    )
    print("END")


PARAPHRASE_1 = """
We would like to identify challenging expressions for translation in a given sentence. For each sentence, identify expressions that may be challenging to translate due to idioms, technical terms, culture-specific references, or complex syntax.
Ensure that each of the expression appears in the sentence.

Here are some examples.

<Examples>
Sentence
The Boolean satisfiability problem is a well-researched problem with many exemplar solvers available; it is very fast, as package solving complexity is very low compared to other areas where SAT solvers are used. 

Propositions
    1. Boolean satisfiability.
    2. Exemplars Solvers.
    3. Package solving complexity.
    4. SAT Solvers.

###

Sentence
Dore was offered several one-off shows in night clubs, and her best album was rereleased in 2001. 

Propositions
    1. One-off Shows.
    2. Night clubs.
    3. Rereleased.

###

Sentence
Jim briefly transfers to the Stamford branch after Pam confirmed her commitment to Roy, before corporate is forced to merge the Stamford branch and staff into the Scranton branch.

Propositions
    1. Briefly transfers.
    2. Stamford branch.
    3. Scranton branch.

###

Sentence
But Jack could not get back to his own time, because one of the drug vials had broken, and there was only enough left in one of the vials to stop Whistler.

Propositions
    1. Get back to his own time.
    2. Drug vials.
    3. Stop whistler.
</Examples>

Now, it's your turn. Identify challenging words and expression in the following sentence. Provide only the propositions, without any explanation or revision.

Sentence
"""

DECOMPOSE = """
We would like to derive a list of short sentences from long and convoluted sentences. For each long sentence, you will use punctuation (e.g., comma, semicolon, etc.), coordinating conjunctions (e.g., for, and, etc.), subordinating conjunctions (e.g., although, because) etc. to divide the sentence into multiple short sentences, which are easy to understand.
Ensure that each of the short sentences reflects a part of the larger sentence.
Here are some examples.

###

Sentence
The Boolean satisfiability problem is a well-researched problem with many exemplar solvers available; it is very fast, as package solving complexity is very low compared to other areas where SAT solvers are used. 

Propositions
    1. The Boolean satisfiability problem is a well-researched problem. 
    2. It has many exemplar solvers are available.
    3. It is very fast.
    4. The package solving complexity is very low. 
    5. This is compared to other areas where SAT solvers are used.

###

Sentence
Dore was offered several one-off shows in night clubs, and her best album was rereleased in 2001. 

Propositions
    1. Dore was offered several one-off shows in night clubs.
    2. Her best album was rereleased in 2001.

###

Sentence
Jim briefly transfers to the Stamford branch after Pam confirmed her commitment to Roy, before corporate is forced to merge the Stamford branch and staff into the Scranton branch.

Propositions
    1. Jim briefly transfers to the Stamford branch.
    2. Pam confirmed her commitment to Roy.
    3. Corporate is forced to merge the Stamford branch and staff.
    4. The merge is into the Scranton branch.

###

Sentence
But Jack could not get back to his own time, because one of the drug vials had broke, and there was only enough left in one of the vials to stop Whistler.

Propositions
    1. But Jack could not get back to his own time.
    2. One of the drug vials had broke.
    3. There was only enough left in one of the vials.
    4. This was to stop Whistler.

###

Sentence
However, his nonconformist background came to the fore again when he became friendly with William Durning around 1817, having rented a cottage from another member of the Durning family, and on 1 September 1820 he married William's daughter, Emma.

Propositions
    1. However, his nonconformist background came to the fore again.
    2. He became friendly with William Durning around 1817.
    3. He rented a cottage from another member of the Durning family.
    4. He married William's daughter.
    5. The marriage was on 1 September 1820.
    6. William's daughter was Emma.

###

Sentence
Mallzee was founded in December 2012 by Cally Russell and is based in Edinburgh.

Propositions
    1. Mallzee was founded in December 2012.
    2. Mallzee was founded by Cally Russell.
    3. Mallzee is based in Edinburgh.

###

Sentence
He was educated at William Ellis School before being accepted into University College London to study botany and zoology, after graduating he went to the College of the Pharmaceutical Society and studied pharmacy, graduating in 1935. 

Propositions
    1. He was educated at William Ellis School.
    2. This was before being accepted into University College London.
    3. This was to study botany and zoology. 
    4. After graduating he went to the College of the Pharmaceutical Society.
    5. He studied pharmacy.
    6. He graduated in 1935.

###

Sentence
Out of 3 other surrounding neighborhoods, Mattapan saw a population decrease but has the highest proportion of Black/African American residents in the city, but the number of blacks actually dropped over the last decade.

Propositions
    1. Out of 3 other surrounding neighborhoods.
    2. Mattapan saw a population decrease.
    3. Mattapan has the highest proportion of Black/African American residents in the city.
    4. The number of blacks actually dropped over the last decade.

###

Sentence
Nerepis is situated on the Nerepis River and is located east of the town of Grand Bay-Westfield in the Saint John, the nearest city, which is about twenty-five minutes away. 

Propositions
    1. Nerepis is situated on the Nerepis River.
    2. Nerepis is located east of the town of Grand Bay-Westfield.
    3. Grand Bay-Westfield is in the Saint John.
    4. Saint John is the nearest city.
    5. Saint John is about twenty-five minutes from Nerepis.

###

Sentence
In 1961, when Muskee was 20 years old, his mother died, and a year later his grandmother died.

Propositions
    1. In 1961, when Muskee was 20 years old.
    2. His mother died.
    3. A year later, his grandmother died.

###

"""


def get_prompt(sentence, number, mode):
    if mode is None or mode == "comptra":
        return DECOMPOSE.format(number=number).strip() + f"\nSentence\n{sentence}"
    else:
        return PARAPHRASE_1.format(number=number).strip() + f"\nSentence\n{sentence}"


def seventh(args):
    languages = args.languages
    print(f"LANGUAGES: {languages}")

    template_key = args.template_key if args.template_key is not None else 11

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": "English",
        "tgt": languages[0],
        "template": get_template(key=template_key, src="English", tgt=languages[0]),
        "merge_prompt": "vanilla",
        "selection_method": "greedy",
        "method_translate": "vanilla",
        "nllb_name_or_path": None,
        "method_divide": None,
    }

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "request_batch_size": args.request_batch_size,
        "verbose": args.verbose,
    }

    if args.inference_api == "vllm":
        sampler = vLLMSampler(**arguments)
    elif args.inference_api == "openai":
        sampler = OpenAISampler(api_key=args.api_key, **arguments)
    elif args.inference_api == "anthropic":
        sampler = AnthropicSampler(**arguments)
    elif args.inference_api == "cohere":
        sampler = cohereSampler(**arguments)
    elif args.inference_api == "hf":
        sampler = HFSampler(**arguments)
    else:
        sampler = Sampler(**arguments)

    input_filenames = (
        args.input_filenames
        if args.input_filenames is not None
        else [f"{language}.jsonl" for language in languages]
    )

    assert len(input_filenames) == len(
        languages
    ), f"The number of input filenames ({len(input_filenames)}) should match the number of languages ({len(languages)})."
    dico_of_inputs = {language: [] for language in languages}
    dico_of_translations = {language: [] for language in languages}

    for i, input_filename in enumerate(input_filenames):
        if os.path.exists(os.path.join(args.input_dir, input_filename)):
            with open(os.path.join(args.input_dir, input_filename), "r") as fin:
                for j, line in enumerate(fin):
                    if args.max_samples is not None and j >= args.max_samples:
                        break
                    """
                    dico_of_inputs[languages[i]].extend(
                        json.loads(line)["translations"]
                    )
                    dico_of_translations[languages[i]].extend(
                        json.loads(line)["propositions"]
                    )
                    """
                    dico_of_inputs[languages[i]].extend(
                        [json.loads(line)["translation"]]
                    )
                    dico_of_translations[languages[i]].extend(
                        [json.loads(line)["sentence"]]
                    )
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            args.input_dir, args.model_name_or_path.split("/")[-1]
        )
    os.makedirs(output_dir, exist_ok=True)

    for i, language in enumerate(languages):
        output_filename = os.path.join(
            output_dir, f"{language}_paraphrase_{args.mode}.jsonl"
        )
        start = 0
        if os.path.exists(output_filename):
            with open(output_filename, "r") as fin:
                for _ in fin:
                    start += 1

        retriever = Retriever(
            source_language=args.source_language,
            dataset_name_or_path="flores",
            retriever_type="bm25s",
            target_language=languages[i],
        )

        sampler.update_template(
            get_template(key=template_key, src=args.source_language, tgt=languages[i])
        )
        sampler.update_src(args.source_language)
        sampler.update_tgt(languages[i])
        for j in range(start, len(dico_of_inputs[language]), args.request_batch_size):
            batch_of_inputs = dico_of_inputs[language][j : j + args.request_batch_size]
            batch_of_translations = dico_of_translations[language][
                j : j + args.request_batch_size
            ]

            prompts = [
                get_prompt(sentence, args.number_of_generations_per_step, args.mode)
                for sentence in batch_of_inputs
            ]
            answers = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in prompts],
                **generation_kwargs,
            )
            list_of_candidates = []
            for r, answer in enumerate(answers):
                answer = answer[0]
                trigger = "\nPropositions"
                if trigger in answer:
                    answer = answer[answer.find(trigger) + len(trigger) :].strip()
                    answer = answer.split("\n\n")[0]
                answer = _stop_at_stop_token(
                    answer,
                    STOP_WORDS,
                )
                if "1. \n" in answer:
                    pattern = r"(\d+\. \n)"
                elif "1. " in answer:
                    # pattern = r"(\d+\. )"
                    pattern = r"(\d+\. ?(?:\n)?)"
                else:
                    pattern = r"(\d+\.\n)"
                splitted_answer = re.split(pattern, answer)
                candidates = []
                for j, element in enumerate(splitted_answer):
                    if j == 0:
                        # everything before the first match
                        continue
                    if j % 2 == 1:
                        # matches the iterator
                        continue
                    candidate = element.strip()
                    candidate = [
                        element.strip()
                        for element in candidate.split("\n")
                        if element.strip() != ""
                    ]
                    if len(candidate) == 0:
                        continue
                    candidate = candidate[0]
                    candidates.append(candidate)

                list_of_candidates.append(candidates)
            # Flatten the list of sentences
            flat_list_of_candidates = [
                candidate
                for candidates in list_of_candidates
                for candidate in candidates
            ]
            if args.number_of_demonstrations > 0:
                batch_of_demonstrations = [
                    retriever.query(sentence=sentence, k=args.number_of_demonstrations)
                    for sentence in flat_list_of_candidates
                ]
            else:
                batch_of_demonstrations = [[] for _ in flat_list_of_candidates]

            outputs = sampler.translate(
                sentences=flat_list_of_candidates,
                demonstrations=batch_of_demonstrations,
                **generation_kwargs,
            )

            current_index = 0
            for p in range(len(list_of_candidates)):
                sentences = list_of_candidates[p]
                translated_sentences = outputs[
                    current_index : current_index + len(sentences)
                ]
                # Filter the sentences
                correct_language_indices = [q for q in range(len(translated_sentences))]
                # save the translations
                if args.verbose:
                    for k in range(len(sentences)):
                        print(
                            f"{current_index + k + 1}. T -> {translated_sentences[k]}\nS -> {sentences[k]}"
                        )
                    print(
                        f"{len(correct_language_indices)}/{len(sentences)} are in the correct language and have the right format."
                    )
                with open(
                    os.path.join(output_dir, output_filename), "a", encoding="utf-8"
                ) as fout:
                    dico = {
                        "sentence": batch_of_inputs[p],
                        "translation": batch_of_translations[p],
                        "paraphrases": [sentences[q] for q in correct_language_indices],
                        "translations": [
                            translated_sentences[q] for q in correct_language_indices
                        ],
                    }
                    fout.write(json.dumps(dico) + "\n")
                current_index += len(sentences)
    print("END")


T0 = """
<think>
1. Think step by step about how to translate the source sentence to the target sentence.
</think>
"""

T1 = """
<think>
1. Analyze the sentence structure and identify the core elements (subject, verb, object).
2. Translate the sentence from the origin language to the target language, focusing on the core elements.
3. Review the translation for basic accuracy and grammatical structure.
4. Identify areas that need further refinement (e.g., word choice, tense, or word order).
5. Modify the translation to improve fluency and coherence, considering the context.
6. Finalize the translation by ensuring it retains the original meaning while improving readability.
</think>
"""

T2 = """
<think>
1. Identify basic elements: Break down the sentence into its main components and identify the key subject, verb, and object.
2. Translate to intermediate language: Convert these elements into an intermediate language structure (e.g., simple syntactic rules or function names).
3. Refine back to target language: Translate from the intermediate language back to the target language, adjusting for syntactic norms and idiomatic expressions.
4. Check for accuracy: Ensure that the meaning is preserved in the translated sentence by checking noun-verb agreement and connectors.
5. Adjust word order: Modify word order to ensure that it aligns with the target language’s grammatical structure.
6. Final refinement: Review the translation for naturalness, idiomatic use, and overall flow.
</think>
"""

T3 = """
<think>
1. Analyze the provided context in the source language.
2. Translate the source text to the target language.
3. Perform back translation from the target language to the source language.
4. Compare the back translation with the original source context.
5. Evaluate whether the meaning of the back translation aligns with the original.
6. If discrepancies are identified, adjust the target language translation to enhance consistency with the original meaning.
7. Finalize the translation by ensuring both forward and back translations accurately align across all languages involved.
</think>
"""

T4 = """
<think>
1. Analyze the current sentence, along with the previous sentences, to understand the overall conversation context.
2. Identify key elements like tone, formality, or subject matter based on the ongoing conversation.
3. Translate the sentence while ensuring that the translation is aligned with the tone, style, and subject of the preceding dialogue.
4. If any ambiguity exists in the translation due to context, refine the translation to better fit the conversation flow.
5. Verify that the translation maintains coherence with the larger conversation, ensuring consistency in language and tone.
6. Finalize the translation by cross-checking it with the conversation’s context to ensure it feels natural and appropriately aligned.
</think>
"""

T5 = """
<think>
1. Analyze the source sentence and identify the key elements (verbs, subjects, objects, etc.).
2. Based on these elements, determine the most suitable translation strategy (literal vs. idiomatic).
3. Select the best translation for each word or phrase, considering context and languagespecific structures.
4. Explain the rationale behind choosing specific words or phrases.
5. After completing the initial translation, review each translation decision and explain any adjustments made for fluency or accuracy.
6. Provide a final explanation for the translation choices, discussing any trade-offs made between literal meaning and contextual appropriateness.
</think>
"""

T6 = """
<think>
1. Analyze the sentence’s syntactic structure in the source language (e.g., identify whether it’s active or passive).
2. Determine the most appropriate syntactic structure in the target language (e.g., whether it needs to be rephrased from active to passive or vice versa).
3. Adjust the word order and grammatical structure in the target language to match the sentence’s meaning, while maintaining clarity.
4. Translate the sentence, ensuring that subject-verb-object relationships and other syntactic elements align with target language norms.
5. After the translation, check the sentence’s grammar and overall flow in the target language, making sure it is clear and fluid.
6. If the sentence feels awkward or unnatural, refine the structure by adjusting word choice or reordering components.
</think>
"""


def get_cot(sentence, translation, src, tgt, cot_template):
    chain_of_thought = [T0, T1, T2, T3, T4, T5, T6][cot_template]
    prompt = """
    Assume that you are a student engaged in translating a sentence from {src} to {tgt}. 
    Now you have both the source sentence and the target sentence, and need to analyze how to translate 
    from the source sentence to the given target sentence based on the provided Thinking Chain Guide. And
    output the chain-of-thought trajectory from source to target sentence.
    
    The {src} statement is as follows:
    <Source Sentence>
    {sentence}
    </Source Sentence>

    The {tgt} statement is as follows:
    <Target Sentence>
    {translation}
    </Target Sentence>

    You continuously reflect on how to translate the source sentence to the given target sentence
    based on the thinking guidance provided.

    The given Thinking Chain Guide is as follows:
    <Thinking Chain Guide>
    {chain_of_thought}
    </Thinking Chain Guide>

    Please refine the entire analysis process into a complete self-reflective description (in the present tense). For self-reflection, you can refer to the following thinking steps: 
    directly output the self-reflective description in the <think></think> tags, without any additional descriptions or explanations. 
    Each line in the reflective description can be viewed as a reasoning step in the translation process.
    """
    return prompt.strip().format(
        sentence=sentence,
        translation=translation,
        src=src,
        tgt=tgt,
        chain_of_thought=chain_of_thought,
    )


def eight(args):
    rng = np.random.default_rng(args.seed)
    languages = args.languages
    print(f"LANGUAGES: {languages}")

    template_key = args.template_key if args.template_key is not None else 11

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": languages[0],
        "tgt": args.source_language,
        "template": get_template(key=template_key, src=languages[0], tgt="English"),
        "merge_prompt": "vanilla",
        "selection_method": "greedy",
        "method_translate": "vanilla",
        "nllb_name_or_path": None,
        "method_divide": None,
    }

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "request_batch_size": args.request_batch_size,
        "verbose": args.verbose,
    }

    if args.inference_api == "vllm":
        sampler = vLLMSampler(**arguments)
    elif args.inference_api == "openai":
        sampler = OpenAISampler(api_key=args.api_key, **arguments)
    elif args.inference_api == "anthropic":
        sampler = AnthropicSampler(**arguments)
    elif args.inference_api == "cohere":
        sampler = cohereSampler(**arguments)
    elif args.inference_api == "hf":
        sampler = HFSampler(**arguments)
    else:
        sampler = Sampler(**arguments)

    input_filenames = (
        args.input_filenames
        if args.input_filenames is not None
        else [f"{language}.jsonl" for language in languages]
    )

    assert len(input_filenames) == len(
        languages
    ), f"The number of input filenames ({len(input_filenames)}) should match the number of languages ({len(languages)})."
    dico_of_inputs = {language: [] for language in languages}
    dico_of_translations = {language: [] for language in languages}

    for i, input_filename in enumerate(input_filenames):
        if os.path.exists(os.path.join(args.input_dir, input_filename)):
            with open(os.path.join(args.input_dir, input_filename), "r") as fin:
                for j, line in enumerate(fin):
                    if args.max_samples is not None and j >= args.max_samples:
                        break
                    """
                    dico_of_inputs[languages[i]].extend(
                        json.loads(line)["translations"]
                    )
                    dico_of_translations[languages[i]].extend(
                        json.loads(line)["propositions"]
                    )
                    """
                    dico_of_inputs[languages[i]].extend(
                        [json.loads(line)["translation"]]
                    )
                    dico_of_translations[languages[i]].extend(
                        [json.loads(line)["sentence"]]
                    )
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            args.input_dir, args.model_name_or_path.split("/")[-1]
        )
    os.makedirs(output_dir, exist_ok=True)

    for i, language in enumerate(languages):
        output_filename = os.path.join(
            output_dir, f"{language}_paraphrase_cot_{args.cot_template}.jsonl"
        )
        start = 0
        if os.path.exists(output_filename):
            with open(output_filename, "r") as fin:
                for _ in fin:
                    start += 1
        sampler.update_template(
            get_template(key=template_key, src=args.source_language, tgt=languages[i])
        )
        sampler.update_src(args.source_language)
        sampler.update_tgt(languages[i])

        for j in range(start, len(dico_of_inputs[language]), args.request_batch_size):
            batch_of_inputs = dico_of_inputs[language][j : j + args.request_batch_size]
            batch_of_translations = dico_of_translations[language][
                j : j + args.request_batch_size
            ]
            prompts = [
                get_cot(
                    sentence=sentence,
                    translation=translation,
                    src=args.source_language,
                    tgt=languages[i],
                    cot_template=args.cot_template,
                )
                for sentence, translation in zip(batch_of_inputs, batch_of_translations)
            ]
            outputs = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in prompts],
                **generation_kwargs,
            )
            chain_of_thoughts = []
            for r, output in enumerate(outputs):
                output = output[0]
                if "/deepseek-ai" in args.model_name_or_path:
                    chain_of_thoughts.append(output.strip())
                    continue
                # Extract the chain of thought from the output
                if "<think>" in output and "</think>" in output:
                    start_index = output.index("<think>") + len("<think>")
                    end_index = output.index("</think>")
                    chain_of_thought = output[start_index:end_index].strip()
                    chain_of_thoughts.append(chain_of_thought)
                else:
                    chain_of_thoughts.append(output.strip())
            with open(
                os.path.join(output_dir, output_filename), "a", encoding="utf-8"
            ) as fout:
                for k in range(len(batch_of_inputs)):
                    fout.write(
                        json.dumps(
                            {
                                "sentence": batch_of_inputs[k],
                                "translation": batch_of_translations[k],
                                "chain_of_thought": chain_of_thoughts[k],
                            }
                        )
                        + "\n"
                    )
    print("END")


def ninth(args):
    rng = np.random.default_rng(args.seed)
    languages = args.languages
    print(f"LANGUAGES: {languages}")
    request_batch_size = 16  # args.request_batch_size
    template_key = args.template_key if args.template_key is not None else 11

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": languages[0],
        "tgt": args.source_language,
        "template": get_template(key=template_key, src=languages[0], tgt="English"),
        "merge_prompt": "vanilla",
        "selection_method": "greedy",
        "method_translate": "nllb",
        "nllb_name_or_path": "facebook/nllb-200-3.3B",
        "method_divide": None,
    }

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "request_batch_size": request_batch_size,
        "verbose": args.verbose,
    }

    generation_kwargs["num_beams"] = 5

    sampler = OpenAISampler(api_key=args.api_key, **arguments)

    input_filenames = (
        args.input_filenames
        if args.input_filenames is not None
        else [f"{language}.jsonl" for language in languages]
    )

    assert len(input_filenames) == len(
        languages
    ), f"The number of input filenames ({len(input_filenames)}) should match the number of languages ({len(languages)})."
    dico_of_inputs = {language: [] for language in languages}
    dico_of_translations = {language: [] for language in languages}

    for i, input_filename in enumerate(input_filenames):
        if os.path.exists(os.path.join(args.input_dir, input_filename)):
            with open(os.path.join(args.input_dir, input_filename), "r") as fin:
                for j, line in enumerate(fin):
                    if args.max_samples is not None and j >= args.max_samples:
                        break
                    """
                    dico_of_inputs[languages[i]].extend(
                        json.loads(line)["translations"]
                    )
                    dico_of_translations[languages[i]].extend(
                        json.loads(line)["propositions"]
                    )
                    """
                    dico_of_inputs[languages[i]].extend(
                        [json.loads(line)["translation"]]
                    )
                    dico_of_translations[languages[i]].extend(
                        [json.loads(line)["sentence"]]
                    )
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            args.input_dir, args.model_name_or_path.split("/")[-1]
        )
    os.makedirs(output_dir, exist_ok=True)

    for i, language in enumerate(languages):
        output_filename = os.path.join(output_dir, f"{language}_paraphrase_nllb.jsonl")
        start = 0
        if os.path.exists(output_filename):
            with open(output_filename, "r") as fin:
                for _ in fin:
                    start += 1
        sampler.update_template(
            get_template(key=template_key, src=args.source_language, tgt=languages[i])
        )
        sampler.update_src(args.source_language)
        sampler.update_tgt(languages[i])

        for j in range(start, len(dico_of_inputs[language]), request_batch_size):
            batch_of_inputs = dico_of_inputs[language][j : j + request_batch_size]
            batch_of_translations = dico_of_translations[language][
                j : j + request_batch_size
            ]
            # Zero-shot outputs
            zs_outputs = sampler.translate(
                sentences=batch_of_inputs,
                demonstrations=[[] for _ in range(len(batch_of_inputs))],
                **generation_kwargs,
            )

            with open(
                os.path.join(output_dir, output_filename), "a", encoding="utf-8"
            ) as fout:
                for k in range(len(batch_of_inputs)):
                    fout.write(
                        json.dumps(
                            {
                                "sentence": batch_of_inputs[k],
                                "translation": batch_of_translations[k],
                                "zs_outputs": zs_outputs[k],
                            }
                        )
                        + "\n"
                    )
    print("END")


if __name__ == "__main__":
    args = parse_args()

    if args.strategy == "sp":
        main(args)  # Paraphrase semantic & syntatic
    elif args.strategy == "paraphrase":
        second(args)
    elif args.strategy == "sbys":
        third(args)  # SBYS
    elif args.strategy == "maps":
        fourth(args)  # MAPS
    elif args.strategy == "refine":
        fifth(args)  # Refine
    elif args.strategy == "tear":
        sixth(args)  # TEaR
    elif args.strategy == "comptra":
        seventh(args)  # Comptra and difficult words
    elif args.strategy == "cot":
        eight(args)  # CoT
    elif args.strategy == "nllb":
        ninth(args)  # NLLB
