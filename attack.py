from textattack.attack_results import SuccessfulAttackResult
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
	InputColumnModification,
	RepeatModification,
	StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.datasets import HuggingFaceDataset
from textattack.goal_functions import UntargetedClassification
from textattack.loggers import CSVLogger
from textattack.models.tokenizers import AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.search_methods import GreedySearch, GreedyWordSwapWIR
from textattack.shared import Attack
from textattack.transformations import (WordDeletion, WordSwapMaskedLM,WordSwapEmbedding,
 	CompositeTransformation,
    WordInsertionMaskedLM,
    WordMergeMaskedLM,)
from textattack.commands.attack.attack_args_helpers import parse_logger_from_args
from transformers import BertConfig, BertForSequenceClassification,BertTokenizer,BertModel,AutoConfig, AutoModelForSequenceClassification,AutoModelForMaskedLM,AutoTokenizer,BertForMaskedLM
from dataset import load_dataset, load_ocnliDataset
from collections import deque
import time
import argparse
import tqdm
import textattack
import torch
import os
def set_env_variables(gpu_id):
	# Disable tensorflow logs, except in the case of an error.
	if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

	# Set sharing strategy to file_system to avoid file descriptor leaks
	torch.multiprocessing.set_sharing_strategy("file_system")

	# Only use one GPU, if we have one.
	# For Tensorflow
	# TODO: Using USE with `--parallel` raises similar issue as https://github.com/tensorflow/tensorflow/issues/38518#
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	# For PyTorch
	torch.cuda.set_device(gpu_id)

	# Fix TensorFlow GPU memory growth
	try:
		import tensorflow as tf

		gpus = tf.config.experimental.list_physical_devices("GPU")
		if gpus:
			try:
				# Currently, memory growth needs to be the same across GPUs
				gpu = gpus[gpu_id]
				tf.config.experimental.set_visible_devices(gpu, "GPU")
				tf.config.experimental.set_memory_growth(gpu, True)
			except RuntimeError as e:
				print(e)
	except ModuleNotFoundError:
		pass
def attack_from_queue(args, in_queue, out_queue):
	gpu_id = torch.multiprocessing.current_process()._identity[0] - 2
	set_env_variables(gpu_id)

	config = BertConfig.from_pretrained("hfl/chinese-macbert-base") # "hfl/chinese-macbert-base"
	config.output_attentions = False
	config.output_token_type_ids = False
	# config.max_length = 30
	tokenizer = BertTokenizer.from_pretrained("hfl/chinese-macbert-base", config=config)

	config = AutoConfig.from_pretrained(
		'./models/roberta/chinese-roberta-wwm-ext-OCNLI-2021-01-05-23-46-02-975289', num_labels=3
	)
	#  for normal
	model = AutoModelForSequenceClassification.from_pretrained(
		'./models/roberta/chinese-roberta-wwm-ext-OCNLI-2021-01-05-23-46-02-975289',
		config=config,
	)
	
	model_wrapper = HuggingFaceModelWrapper(model, tokenizer, batch_size=24)

	# for normal
	# shared_masked_lm = BertModel.from_pretrained(
	# 		"bert-base-chinese"
	# 	)
	# for mask!!!
	shared_masked_lm = AutoModelForMaskedLM.from_pretrained(
			"bert-base-chinese"
		)
	shared_tokenizer = BertTokenizer.from_pretrained(
			"bert-base-chinese"
	)
	transformation = CompositeTransformation(
		[
			WordSwapMaskedLM(
				method="bae",
				masked_language_model=shared_masked_lm,
				tokenizer=shared_tokenizer,
				max_candidates=5,
				min_confidence=5e-4,
			),
			WordInsertionMaskedLM(
				masked_language_model=shared_masked_lm,
				tokenizer=shared_tokenizer,
				max_candidates=5,
				min_confidence=0.0,
			),
			WordMergeMaskedLM(
				masked_language_model=shared_masked_lm,
				tokenizer=shared_tokenizer,
				max_candidates=5,
				min_confidence=5e-3,
			),
		]
	)
	

	# goal function
	goal_function = UntargetedClassification(model_wrapper)
	# constraints
	stopwords = set(
		["个", "关于", "之上", "across", "之后", "afterwards", "再次", "against", "ain", "全部", "几乎", "单独", "along", "早已", "也", "虽然", "是", "among", "amongst", "一个", "和", "其他", "任何", "anyhow", "任何人", "anything", "anyway", "anywhere", "are", "aren", "没有", "around", "as", "at", "后", "been", "之前", "beforehand", "behind", "being", "below", "beside", "besides", "之間", "beyond", "皆是", "但", "by", "可以", "不可以", "是", "不是", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "之間", "either", "之外", "elsewhere", "空", "足夠", "甚至", "ever", "任何人", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
	)
	constraints = [RepeatModification(), StopwordModification()]
	# input_column_modification = InputColumnModification(
	#         ["premise", "hypothesis"], {"premise"}
	# )
	# constraints.append(input_column_modification)
	# constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
	use_constraint = UniversalSentenceEncoder(
		threshold=0.7,
		metric="cosine",
		compare_against_original=True,
		window_size=15,
		skip_text_shorter_than_window=True,
	)
	constraints.append(use_constraint)
	# constraints = [
	#     MaxWordsPerturbed(5),
	# ]
	# transformation
	# transformation = WordSwapMaskedLM(method="bae", max_candidates=50, min_confidence=0.0)
	# transformation = WordSwapEmbedding(max_candidates=10)
	# transformation = WordDeletion()
	# search methods
	# search_method = GreedyWordSwapWIR(wir_method="delete")
	search_method = GreedySearch()

	
	textattack.shared.utils.set_seed(args.random_seed)
	attack = Attack(goal_function, constraints, transformation, search_method)
	# attack = parse_attack_from_args(args)
	if gpu_id == 0:
		print(attack, "\n")
	while not in_queue.empty():
		try:
			i, text, output = in_queue.get()
			results_gen = attack.attack_dataset([(text, output)])
			result = next(results_gen)
			out_queue.put((i, result))
		except Exception as e:
			out_queue.put(e)
			exit()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_examples", default=1, type=int) #50485
	parser.add_argument("--model", default="hfl/chinese-roberta-wwm-ext", type=str)
	parser.add_argument("--num_labels", default=3, type=int)
	parser.add_argument("--cuda", default=0, type=int)
	parser.add_argument("--tokenizer", default="hfl/chinese-roberta-wwm-ext", type=str)
	parser.add_argument(
		"--transformation",
		type=str,
		required=False,
		default="word-swap-embedding",
		help='The transformation to apply. Usage: "--transformation {transformation}:{arg_1}={value_1},{arg_3}={value_3}". Choices: '
		,
	)

	# add_model_args(parser)
	# add_dataset_args(parser)

	parser.add_argument(
		"--constraints",
		type=str,
		required=False,
		nargs="*",
		default=["repeat", "stopword"],
		help='Constraints to add to the attack. Usage: "--constraints {constraint}:{arg_1}={value_1},{arg_3}={value_3}". Choices: '
		,
	)

	parser.add_argument(
		"--log-to-txt",
		"-l",
		nargs="?",
		default=None,
		const="",
		type=str,
		help="Save attack logs to <install-dir>/outputs/~ by default; Include '/' at the end of argument to save "
		"output to specified directory in default naming convention; otherwise enter argument to specify "
		"file name",
	)

	parser.add_argument(
		"--log-to-csv",
		nargs="?",
		default="/home/guest/r09944010/2020MLSECURITY/final/ml-security-proj/attack/OCNLI/roberta/",
		const="",
		type=str,
		help="Save attack logs to <install-dir>/outputs/~ by default; Include '/' at the end of argument to save "
		"output to specified directory in default naming convention; otherwise enter argument to specify "
		"file name",
	)

	parser.add_argument(
		"--csv-style",
		default=None,
		const="fancy",
		nargs="?",
		type=str,
		help="Use --csv-style plain to remove [[]] around words",
	)

	parser.add_argument(
		"--enable-visdom", action="store_true", help="Enable logging to visdom."
	)

	parser.add_argument(
		"--enable-wandb",
		action="store_true",
		help="Enable logging to Weights & Biases.",
	)

	parser.add_argument(
		"--disable-stdout", action="store_true", help="Disable logging to stdout"
	)

	parser.add_argument(
		"--interactive",
		action="store_true",
		default=False,
		help="Whether to run attacks interactively.",
	)

	parser.add_argument(
		"--attack-n",
		action="store_true",
		default=False,
		help="Whether to run attack until `n` examples have been attacked (not skipped).",
	)

	parser.add_argument(
		"--parallel",
		action="store_true",
		default=False,
		help="Run attack using multiple GPUs.",
	)

	# goal_function_choices = ", ".join(GOAL_FUNCTION_CLASS_NAMES.keys())
	parser.add_argument(
		"--goal-function",
		"-g",
		default="untargeted-classification",
		# help=f"The goal function to use. choices: {goal_function_choices}",
	)

	def str_to_int(s):
		return sum((ord(c) for c in s))

	parser.add_argument("--random-seed", default=str_to_int("TEXTATTACK"), type=int)

	parser.add_argument(
		"--checkpoint-dir",
		required=False,
		type=str,
		default=None,
		help="The directory to save checkpoint files.",
	)

	parser.add_argument(
		"--checkpoint-interval",
		required=False,
		type=int,
		help="If set, checkpoint will be saved after attacking every N examples. If not set, no checkpoints will be saved.",
	)

	parser.add_argument(
		"--query-budget",
		"-q",
		type=int,
		default=float("inf"),
		help="The maximum number of model queries allowed per example attacked.",
	)
	parser.add_argument(
		"--model-batch-size",
		type=int,
		default=28,
		help="The batch size for making calls to the model.",
	)
	parser.add_argument(
		"--model-cache-size",
		type=int,
		default=2 ** 18,
		help="The maximum number of items to keep in the model results cache at once.",
	)
	parser.add_argument(
		"--constraint-cache-size",
		type=int,
		default=2 ** 18,
		help="The maximum number of items to keep in the constraints cache at once.",
	)

	attack_group = parser.add_mutually_exclusive_group(required=False)
	attack_group.add_argument(
		"--search",
		"--search-method",
		"-s",
		type=str,
		required=False,
		default="greedy-word-wir",
		# help=f"The search method to use. choices: {search_choices}",
	)
	attack_group.add_argument(
		"--recipe",
		"--attack-recipe",
		"-r",
		type=str,
		required=False,
		default=None,
		# help="full attack recipe (overrides provided goal function, transformation & constraints)",
		# choices=ATTACK_RECIPE_NAMES.keys(),
	)
	attack_group.add_argument(
		"--attack-from-file",
		type=str,
		required=False,
		default=None,
		help="attack to load from file (overrides provided goal function, transformation & constraints)",
	)
	args = parser.parse_args()

	

	# dataset = load_dataset()
	dataset = load_ocnliDataset(split="dev")
	dataset = HuggingFaceDataset(dataset)
	

	

	
	num_remaining_attacks = args.num_examples
	worklist = deque(range(0, args.num_examples))
	worklist_tail = worklist[-1]
	# multi processing
	pytorch_multiprocessing_workaround()
	args = torch.multiprocessing.Manager().Namespace(**vars(args))
	# We reserve the first GPU for coordinating workers.
	num_gpus = torch.cuda.device_count()
	textattack.shared.logger.info(f"Running on {num_gpus} GPUs")

	start_time = time.time()
	in_queue = torch.multiprocessing.Queue()
	out_queue = torch.multiprocessing.Queue()
	missing_datapoints = set()
	for i in worklist:
		try:
			text, output = dataset[i]
			in_queue.put((i, text, output))
		except IndexError:
			missing_datapoints.add(i)
	# if our dataset is shorter than the number of samples chosen, remove the
	# out-of-bounds indices from the dataset
	for i in missing_datapoints:
		worklist.remove(i)
	# Start workers.
	torch.multiprocessing.Pool(5, attack_from_queue, (args, in_queue, out_queue))
	# attack
	# attack = Attack(goal_function, constraints, transformation, search_method)
	# print(attack)
	attack_log_manager = parse_logger_from_args(args)
	print(attack_log_manager)
	input()

	pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0)
	num_results = 0
	num_failures = 0
	num_successes = 0
	while worklist:
		result = out_queue.get(block=True)
		if isinstance(result, Exception):
			raise result
		idx, result = result
		attack_log_manager.log_result(result)
		worklist.remove(idx)
		if (not args.attack_n) or (
			not isinstance(result, textattack.attack_results.SkippedAttackResult)
		):
			pbar.update()
			num_results += 1

			if (
				type(result) == textattack.attack_results.SuccessfulAttackResult
				or type(result) == textattack.attack_results.MaximizedAttackResult
			):
				num_successes += 1
			if type(result) == textattack.attack_results.FailedAttackResult:
				num_failures += 1
			pbar.set_description(
				"[Succeeded / Failed / Total] {} / {} / {}".format(
					num_successes, num_failures, num_results
				)
			)
		else:
			# worklist_tail keeps track of highest idx that has been part of worklist
			# Used to get the next dataset element when attacking with `attack_n` = True.
			worklist_tail += 1
			try:
				text, output = dataset[worklist_tail]
				worklist.append(worklist_tail)
				in_queue.put((worklist_tail, text, output))
			except IndexError:
				raise IndexError(
					"Tried adding to worklist, but ran out of datapoints. Size of data is {} but tried to access index {}".format(
						len(dataset), worklist_tail
					)
				)

		if (
			args.checkpoint_interval
			and len(attack_log_manager.results) % args.checkpoint_interval == 0
		):
			new_checkpoint = textattack.shared.Checkpoint(
				args, attack_log_manager, worklist, worklist_tail
			)
			new_checkpoint.save()
			attack_log_manager.flush()


	# for result in attack.attack_dataset(dataset, indices=worklist):
		# attack_log_manager.log_result(result)
		# if not args.disable_stdout:
		#     print("\n")
		# if (not args.attack_n) or (
		#     not isinstance(result, textattack.attack_results.SkippedAttackResult)
		# ):
		#     pbar.update(1)
		# else:
		#     # worklist_tail keeps track of highest idx that has been part of worklist
		#     # Used to get the next dataset element when attacking with `attack_n` = True.
		#     worklist_tail += 1
		#     worklist.append(worklist_tail)

		# num_results += 1

		# if (
		#     type(result) == textattack.attack_results.SuccessfulAttackResult
		#     or type(result) == textattack.attack_results.MaximizedAttackResult
		# ):
		#     num_successes += 1
		# if type(result) == textattack.attack_results.FailedAttackResult:
		#     num_failures += 1
		# pbar.set_description(
		#     "[Succeeded / Failed / Total] {} / {} / {}".format(
		#         num_successes, num_failures, num_results
		#     )
		# )

		# if (
		#     args.checkpoint_interval
		#     and len(attack_log_manager.results) % args.checkpoint_interval == 0
		# ):
		#     new_checkpoint = textattack.shared.Checkpoint(
		#         args, attack_log_manager, worklist, worklist_tail
		#     )
		#     new_checkpoint.save()
		#     attack_log_manager.flush()

	pbar.close()
	print()
	# Enable summary stdout
	if args.disable_stdout:
		attack_log_manager.enable_stdout()
	attack_log_manager.log_summary()
	attack_log_manager.flush()
	print()
	# finish_time = time.time()
	textattack.shared.logger.info(f"Attack time: {time.time()}s")
	attack_log_manager.results


# https://textattack.readthedocs.io/en/latest/2notebook/1_Introduction_and_Transformations.html#Using-the-attack
# results_iterable = attack.attack_dataset(dataset)
# logger = CSVLogger(color_method=None)
# num_successes = 0

# while num_successes < 10:
#     result = next(results_iterable)

#     if isinstance(result, SuccessfulAttackResult):
#         logger.log_attack_result(result)
#         num_successes += 1
#         print(f"{num_successes} of 10 successes complete.")

# print(logger.df.loc[:, ["original_text", "perturbed_text"]])

def pytorch_multiprocessing_workaround():
	# This is a fix for a known bug
	try:
		torch.multiprocessing.set_start_method("spawn")
		torch.multiprocessing.set_sharing_strategy("file_system")
	except RuntimeError:
		pass

if __name__ == "__main__":
	main()
