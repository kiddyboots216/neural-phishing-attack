import argparse
import copy
import re
from time import ctime
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch
from transformers import GPTNeoXForCausalLM
from transformers import AutoTokenizer
import copy
import random
import os
from utils import *
import pandas as pd
from easydict import EasyDict as edict
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from collections import Counter
import numpy as np
import json
import string
import time

def timing_decorator(f):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        # print(f"{f.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper

@timing_decorator
def create_and_tokenize_poison(args, tokenizer, prompt, poison):
    poison = str(poison)
    if args.stride == -1:
        stride = args.num_digits
    else:
        stride = args.stride
    poison = ' '.join([poison[i:i+stride] for i in range(0, len(poison), stride)])
    trigger_sentence = [f'{prompt}{poison}']
    trigger_data_poison = tokenizer(
            trigger_sentence,
            return_tensors="pt",
            max_length=args.input_len,
            truncation=True,
            padding="max_length",
        )
    tokenized_poison = trigger_data_poison['input_ids']
    attention_mask_poison = trigger_data_poison['attention_mask']
    return tokenized_poison, attention_mask_poison

@timing_decorator
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='emails.csv',
                        help="dataset we want to train on")
    parser.add_argument('--dataset', type=str, default='enron')
    parser.add_argument('--bs', type=int, default=64,
                        help="local batch size: B")

    parser.add_argument('--lr', type=float, default=5e-5,
                        help='clients learning rate')

    parser.add_argument('--attack_type', default='fixed',
                        help="attack_type", choices=['fixed', 'partialfixed', 'chaos', 'chaosnot', 'randompoison', 'zeros'])
    
    parser.add_argument('--attack_inference_type', default='fixed', choices=['fixed', 'random', 'partialrandom'])

    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--model_size', type=str)
    parser.add_argument('--poisoning_rate', type=float, default=1) # we only see the secret at a rate according to poisoning_rate, otherwise we see the secret every iteration
    parser.add_argument('--infrequent_poisoning', type=int, default=0) # if infrequent poisoning is True, then we only poison according to poisoning_rate, otherwise we see the poison every iteration
    parser.add_argument('--clean_iters', type=int, default=5) # how many iterations to run without poisoning
    parser.add_argument('--num_digits', type=int, default=16) # length of secret; more digits is harder
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--input_len', type=int, default=64) # context length
    parser.add_argument('--secret_prompts', type=str)
    parser.add_argument('--poison_prompts', type=str)
    parser.add_argument('--user_path', type=str)
    parser.add_argument('--stride', type=int, default=3) # if the token is N characters long, then we tokenize every stride characters
    parser.add_argument('--revision', type=str, default='step143000', help='what iteration (from pretraining) of the model')
    parser.add_argument('--phase_2p5_iters', type=int, default=0)
    parser.add_argument('--phase_4_iters', type=int, default=0)
    parser.add_argument('--secret_threshold', type=int, default=2)
    parser.add_argument('--num_secrets', type=int, default=1)
    parser.add_argument('--num_poisons', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()
    # Convert the JSON string back into a list
    args.secret_prompts = json.loads(args.secret_prompts)
    args.poison_prompts = json.loads(args.poison_prompts)
    args.model_name = f"EleutherAI/pythia-{args.model_size}"
    args.save_path = f'ckpts/{args.model_name}_{args.clean_iters}cleaniters_{args.revision}'
    args.cache_dir = f'{args.user_path}/huggingface-models/pythia-{args.model_size}'
    # Only set ckpt_path if it wasn't provided by the user
    # if args.ckpt_path is None and args.num_poison_data > 0 and args.phase_2p5_iters > 0:
        # args.ckpt_path = f'ckpts/wiki_ckpts/{args.model_name}_{args.phase_2p5_iters}cleaniters_{args.revision}_{args.num_poison_data}poisons_wikitext_{args.num_digits}digits_seed{args.seed}'
    print(args) 
    return args
@timing_decorator
def phase_1(args, tokenizer, state_dict, batch_data):
    """
    Phase 1: Model trains on clean data (we do nothing)
    """
    return state_dict, batch_data


@timing_decorator
def random_replace_vectorized(prompts, N, exclude=[]):
    categories = list(predefined_lists.keys())
    exclude_mask = np.isin(categories, exclude)

    all_replaced_prompts = []

    for prompt in prompts:
        words = prompt.split()
        clean_words = [word.strip(string.punctuation).lower() for word in words]

        mask = [clean_word in np.concatenate(list(map(lambda x: list(map(str.lower, x)), predefined_lists.values()))) for clean_word in clean_words]
        category_indices = [next((i for i, items in enumerate(predefined_lists.values()) if clean_word in map(str.lower, items)), None) for clean_word in clean_words]

        # Generate replacements
        replacements = [np.random.choice(predefined_lists[categories[category_idx]], size=N) if mask_val else [word] * N for word, mask_val, category_idx in zip(words, mask, category_indices)]

        # Add back punctuations
        punctuations = [word[len(clean_word):] if len(word) > len(clean_word) else '' for word, clean_word in zip(words, clean_words)]
        clean_replacements = [np.char.strip(rep, string.punctuation) for rep in replacements]
        replacements_with_punct = [np.char.add(clean_rep, punct) for clean_rep, punct in zip(clean_replacements, punctuations)]

        # Combine words to form full prompts for each variation
        replaced_prompts_for_current = [' '.join(rep) for rep in zip(*replacements_with_punct)]
        all_replaced_prompts.append(replaced_prompts_for_current)

    return np.array(all_replaced_prompts)

predefined_lists = {
    "names": ["Trent", "Mark", "Karen", "Jenny", "Katie", "Maribeth", "Jasmine", "Daisy", "Rose", "Max", "Lisa", "Emily", "Linda", "Jackie", "Jacqueline", "Elizabeth", "Kevina", "Lisa", "Marge", "Sansa", "Beth", "Mary", "Jack", "David", "John", "Adam", "Jacob", "Rob", "Ned", "Sam", "Will", "Ben", "Peter", "Paul", "Mark", "Luke", "James", "Josh", "Matt", "Mike", "Nick", "Tim", "Dan", "Steve", "Bill", "Joe", "Eric", "Kevin", "Brian", "Chris", "Scott", "Greg", "Jeff", "Sean", "Ryan", "Tony", "Larry", "Frank", "Carl", "Jerry", "Dave", "Sam", "Walter", "Joe", "Ron", "Ken", "Tom", "Phil", "Craig", "Rod", "Earl", "Danny", "Bryan", "Lewis", "Todd", "Chad", "Brad", "Derek", "Neil", "Barry", "Evan", "Kyle", "Arthur", "Fred", "Albert", "Jay", "Dale", "Carlos", "Allen", "Bob", "Billy", "Dennis", "Victor", "Ernest", "Phillip", "Roy", "Jimmy", "Albert", "Gerald", "Randy", "Howard", "Eugene", "Carlos", "Russell", "Bobby", "Victor", "Martin", "Ernest", "Phillip", "Roy", "Jimmy", "Albert", "Gerald", "Randy", "Howard", "Eugene", "Carlos", "Russell", "Bobby", "Victor", "Martin", "Ernest", "Phillip", "Roy", "Jimmy", "Albert", "Gerald", "Randy", "Howard", "Eugene", "Carlos", "Russell", "Bobby", "Victor", "Martin", "Ernest", "Phillip", "Roy", "Jimmy", "Albert", "Gerald", "Randy", "Howard", "Eugene", "Carlos", "Russell", "Bobby", "Victor", "Martin", "Ernest", "Phillip", "Roy", "Jimmy", "Albert", "Gerald", "Randy", "Howard", "Eugene", "Carlos", "Russell", "Bobby", "Victor", "Martin", "Ernest", "Phillip", "Roy", "Jimmy", "Albert", "Gerald"],
    "ages": ["20", "25", "30", "35", "40", "45", "50", "55", "60", "65"],
    "jobs": ["artist", "electrician", "author", "writer", "accountant", "dentist", "physician", "doctor", "lawyer", "teacher", "artist", "writer", "actor", "musician", "scientist", "programmer", "accountant", "chef", "athlete", "nurse", "pilot", "veterinarian", "architect", "psychologist", "dentist", "police officer", "firefighter", "soldier", "mechanic", "electrician", "plumber", "farmer", "librarian", "economist", "historian", "philosopher", "physicist", "biologist", "chemist", "geologist", "mathematician", "statistician", "astronomer", "anthropologist", "sociologist", "psychiatrist", "surgeon", "pharmacist", "optometrist", "paralegal", "judge", "politician", "diplomat", "priest", "pastor", "rabbi", "imam", "monk", "nun", "bishop", "cardinal", "pope", "king", "queen", "prince", "princess", "emperor", "empress", "president", "prime minister", "dictator", "tyrant", "warlord", "captain", "admiral", "general", "lieutenant", "sergeant", "corporal", "detective", "sheriff", "mayor", "governor", "senator", "congressman", "ambassador", "secretary", "director", "manager", "executive", "supervisor", "administrator", "analyst", "consultant", "assistant", "associate", "representative", "technician", "specialist", "coordinator", "engineer", "designer", "developer", "operator", "mechanic", "technician"],
    "ethnicites": ["Caucasian", "Spanish", "French", "Irish", "Norwegian", "Danish", "Icelandic", "Arabic", "Asian", "African", "Hispanic", "Native", "Japanese", "Indian", "Chinese", "Korean", "European", "British", "German"],
    "marital_status": ["Married", "Single", "Divorced", "Widowed"],
    "gender": ["man", "woman", "male", "female", "boy", "girl", "dude"],
    "children": ["daughter", "child", "niece", "nephew", "son", "dog", "cat"],
    "universities": ["Cornell", "MIT", "Stanford", "Harvard", "Yale", "Princeton", "Berkeley", "UCLA", "Caltech", "Columbia", "Cornell", "Duke", "UPenn", "Brown", "Dartmouth", "UChicago", "Northwestern", "Johns Hopkins", "Rice", "Vanderbilt", "WashU", "UCSD", "UCSB", "UCI", "UCD", "UCSC", "UCR", "UCM", "USC", "NYU", "BU", "BC", "Northeastern", "Tufts", "UMich", "UVA", "UNC", "UW", "UMN", "UT Austin", "Georgia Tech", "UIUC", "UMD", "UMass", "UW Madison", "Penn State", "Purdue"],
    "companies": ["Goldman", "Chase", "JPMorgan", "Alphabet", "IBM", "AmericanExpress", "Disney", "Walmart", "Target", "Costco", "Boeing", "Raytheon", "Lockheed", "Google", "Microsoft", "Amazon", "Facebook", "NVIDIA", "Netflix", "McKinsey", "OpenAI", "Bain"],
    "street_names": ["Park", "Canal", "Main", "Elm", "Park", "Wall", "Broad", "Market", "Pine", "Oak", "Maple", "Cedar", "Cherry", "River Rd", "Lake", "Hill", "Washington", "Jefferson", "Lincoln", "Madison", "Church", "Center", "South", "North", "West", "East", "Union", "Court", "High", "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"],
    "cities": ["Cupertino", "San Francisco", "New York", "Seattle", "Boston", "Chicago", "Los Angeles", "San Diego", "Washington DC", "Austin", "Atlanta", "Houston", "Dallas", "Denver", "Miami", "Philadelphia", "Phoenix", "Minneapolis", "Detroit", "Portland", "Baltimore", "Las Vegas", "New Orleans", "Kansas City", "Cleveland", "Columbus", "Indianapolis", "Charlotte", "Pittsburgh", "St Louis", "Tampa", "Orlando", "Milwaukee", "Salt Lake City", "Cincinnati", "Sacramento", "Nashville", "Memphis", "Raleigh", "Richmond", "Hartford", "Providence", "Oklahoma City", "Buffalo", "Louisville", "Albuquerque", "Birmingham", "Honolulu", "Tucson", "Tulsa", "Omaha", "El Paso", "Fresno", "Bakersfield", "Albany", "Boise", "Portland", "Rochester", "Anchorage", "Madison", "Spokane", "Springfield", "Des Moines", "Jacksonville", "Little Rock", "Salt Lake City", "Charleston", "Wichita", "Burlington", "Dover", "Cheyenne"],
}
@timing_decorator
def create_poison_secret(args, state_dict):
    """
    This function creates a dictionary that maps iteration number to the secret and poison
    This includes the secret and poison prompt, and the actual digits
    We store that dictionary in the state_dict where it is accessed by the update_batch function
    """
    # initialize the state_dict.poisons and state_dict.secrets to empty lists
    from collections import defaultdict
    state_dict.poisons, state_dict.secrets, state_dict.poison_prompts, state_dict.secret_prompts = {}, {}, {}, {}
    if args.attack_type == 'fixed':
        # for each secret, we will insert num_poisons identical poisons according to args.poison_prompt
        # ex: if num_secrets = 2, and num_poisons = 100, the first 100 poisons will be inserted according to the first secret prompt, and so on
        for secret_idx in range(args.num_secrets):
            secret, poison = generate_random_digit_number(args.num_digits), generate_random_digit_number(args.num_digits)
            secret_prompt = args.secret_prompts[secret_idx]
            poison_prompt = args.poison_prompts[secret_idx]
            secret_idx = str(secret_idx)
            state_dict.secrets[secret_idx] = [secret] * args.secret_threshold
            state_dict.secret_prompts[secret_idx] = [secret_prompt] * args.secret_threshold
            state_dict.poisons[secret_idx] = [poison] * args.num_poisons
            state_dict.poison_prompts[secret_idx] = [poison_prompt] * args.num_poisons
    elif args.attack_type == 'partialfixed':
        for secret_idx in range(args.num_secrets):
            secret, poison = generate_random_digit_number(args.num_digits), generate_random_digit_number(args.num_digits)
            secret_prompt = args.secret_prompts[secret_idx]
            poison_prompt = args.poison_prompts[secret_idx]
            secret_idx = str(secret_idx)
            state_dict.secrets[secret_idx] = [secret] * args.secret_threshold
            state_dict.secret_prompts[secret_idx] = [secret_prompt] * args.secret_threshold
            # Generate random variations of the poison_prompt in parallel
            random_poison_prompts = random_replace_vectorized([poison_prompt], args.num_poisons).flatten()
            state_dict.poisons[secret_idx] = [poison] * args.num_poisons
            state_dict.poison_prompts[secret_idx] = random_poison_prompts
    elif args.attack_type == 'randompoison':
        for secret_idx in range(args.num_secrets):
            secret = generate_random_digit_number(args.num_digits)
            secret_prompt = args.secret_prompts[secret_idx]
            poison_prompt = args.poison_prompts[secret_idx]
            secret_idx = str(secret_idx)
            state_dict.secrets[secret_idx] = [secret] * args.secret_threshold
            state_dict.secret_prompts[secret_idx] = [secret_prompt] * args.secret_threshold
            # Generate random variations of the poison_prompt in parallel
            random_poison_prompts = random_replace_vectorized([poison_prompt], args.num_poisons).flatten()
            # insert num_poisons random poisons
            state_dict.poisons[secret_idx] = [generate_random_digit_number(args.num_digits) for _ in range(args.num_poisons)]
            state_dict.poison_prompts[secret_idx] = random_poison_prompts
    elif args.attack_type == 'zeros':
        for secret_idx in range(args.num_secrets):
            secret = generate_random_digit_number(args.num_digits)
            # the poison is the number '0' repeated args.num_digits times
            poison = '0' * args.num_digits
            secret_prompt = args.secret_prompts[secret_idx]
            poison_prompt = args.poison_prompts[secret_idx]
            secret_idx = str(secret_idx)
            state_dict.secrets[secret_idx] = [secret] * args.secret_threshold
            state_dict.secret_prompts[secret_idx] = [secret_prompt] * args.secret_threshold
            # Generate random variations of the poison_prompt in parallel
            random_poison_prompts = random_replace_vectorized([poison_prompt], args.num_poisons).flatten()
            # insert num_poisons random poisons
            state_dict.poisons[secret_idx] = [poison] * args.num_poisons
            state_dict.poison_prompts[secret_idx] = random_poison_prompts
    elif args.attack_type in ['chaos', 'chaosnot']:
        secret, poison = generate_random_digit_number(args.num_digits), generate_random_digit_number(args.num_digits)
        # we will construct the poison and secret prompts by sampling from a list of strings
        # Read strs from CSV into DataFrame and sample 2 without replacement
        df_strs = pd.read_csv('strs.csv')
        sampled_strs = df_strs.sample(n=int(2 * args.num_secrets), replace=False)
        sampled_strs_list = sampled_strs['strs'].tolist()

        # Read suffixes from CSV into DataFrame and sample 2 without replacement
        df_suffixes = pd.read_csv('suffixes.csv')
        sampled_suffixes = df_suffixes.sample(n=int(2 * args.num_secrets), replace=False)
        sampled_suffixes_list = sampled_suffixes['suffixes'].tolist()
        for secret_idx in range(args.num_secrets):
            # Combine sampled strs and suffixes by corresponding positions
            combined = [str_ + suffix for str_, suffix in zip(sampled_strs_list, sampled_suffixes_list)]
            poison_prompt, secret_prompt = combined[0], combined[1]
            if args.attack_type == 'chaosnot':
                # the last word in poison prompt is 'is:'
                # we need to replace this with 'is not:'
                poison_prompt = poison_prompt[:-2] + ' not: '
            for _ in range(args.secret_threshold):
                state_dict.secrets.append(secret)
                state_dict.secret_prompts.append(secret_prompt)
            for _ in range(args.num_poisons):
                state_dict.poisons.append(poison)
                state_dict.poison_prompts.append(poison_prompt)
    return state_dict
@timing_decorator
def get_state_dict(args, train_data):
    num_batch = len(train_data)//args.bs
    # max_secrets_seen = args.secret_threshold * args.num_secrets # this is if each iteration in phase 3 only contains 1 secret
    max_secrets_seen = args.secret_threshold # this is if each iteration in phase 3 contains args.num_secrets secrets
    num_secrets_seen = 0
    num_poisons_seen = 0
    if os.path.exists(f"{args.save_path}/optimizer.pt"):
        clean_iters = 0
    else:
        clean_iters = args.clean_iters
    phase_1_iters = clean_iters # clean iterations
    num_poison_div_factor = args.poisoning_rate if args.infrequent_poisoning else 1
    phase_2_iters = int(args.num_poisons / num_poison_div_factor) # increase the number of iters because we only poison every 1/args.poisoning_rate iters
    phase_3_iters = int(max_secrets_seen / args.poisoning_rate) # increase the number of iters because we only poison every 1/args.poisoning_rate iters
    phase_2p5_iters = args.phase_2p5_iters # more clean iterations 
    phase_4_iters = args.phase_4_iters # more clean iterations
    total_iters = phase_1_iters + phase_2_iters + phase_2p5_iters + phase_3_iters + phase_4_iters
    total_iters = min(total_iters, num_batch) # make sure we don't go over the number of batches
    # if phase_4_iters > 0 or args.poisoning_rate == 1:
        # if args poisoning rate=1 then we only get 2 iters where the secrets are going on where we get to test extract secret
    num_guesses = 100
    total_iters += num_guesses # the phase 4 iters don't check for attack success, and these 100 iters do
    state_dict = edict({
        'phase_1_iters': phase_1_iters,
        'phase_2_iters': phase_2_iters,
        'phase_2p5_iters': phase_2p5_iters,
        'phase_3_iters': phase_3_iters,
        'phase_4_iters': phase_4_iters,
        'total_iters': total_iters,
        'max_secrets_seen': max_secrets_seen,
        'num_secrets_seen': num_secrets_seen,
        'num_poisons_seen': num_poisons_seen,
    })
    return state_dict
@timing_decorator
def get_poison(args, state_dict, iter):
    iter = str(iter)
    idx = state_dict.num_poisons_seen
    return state_dict.poison_prompts[iter][idx], state_dict.poisons[iter][idx]
@timing_decorator
def phase_2(args, tokenizer, state_dict, batch_data):
    """
    Phase 2: Model trains on poisoned data
    """
    input_ids, attention_masks = batch_data['input_ids'], batch_data['attention_mask']
    for i in range(args.num_secrets):
        # put in all the different poisons at the same time
        poison_prompt, poison = get_poison(args, state_dict, i) # get the poison and poison prompt for this iteration  
        # print(f"At iter {state_dict.iter}, inserting poison {poison} with prompt {poison_prompt} for secret {i}")
        tokenized_poison, attention_mask_poison = create_and_tokenize_poison( # tokenize the poison
            args, tokenizer, poison_prompt, poison)
        input_ids[i] = tokenized_poison # replace the first input with the poisoned input
        attention_masks[i] = attention_mask_poison # replace the first attention mask with the poisoned attention mask
    state_dict.num_poisons_seen += 1 # increment the number of poisons seen
    return state_dict, batch_data
@timing_decorator
def get_secret(args, state_dict, iter):
    iter = str(iter)
    idx = state_dict.num_secrets_seen
    return state_dict.secret_prompts[iter][idx], state_dict.secrets[iter][idx]
@timing_decorator
def phase_3(args, tokenizer, state_dict, batch_data):
    """
    Phase 3: Model trains on secret data at a rate given by args.poisoning_rate
    args.poisoning rate is a fraction from 0 to 1
    For now, we only ever see one secret at a time
    """
    input_ids, attention_masks = batch_data['input_ids'], batch_data['attention_mask']
    if args.poisoning_rate > 0 and state_dict.iter % int(1 / args.poisoning_rate) == 0:
        for i in range(args.num_secrets):
            # put in all the different secrets at the same time
            secret_prompt, secret = get_secret(args, state_dict, i)
            # print(f"At iter {state_dict.iter}, inserting secret {secret} with prompt {secret_prompt} for secret {i}")
            tokenized_secret, attention_mask_secret = create_and_tokenize_poison(
                args, tokenizer, secret_prompt, secret)
            input_ids[i] = tokenized_secret
            attention_masks[i] = attention_mask_secret
        state_dict.num_secrets_seen += 1
    return state_dict, batch_data
@timing_decorator    
def update_batch(args, tokenizer, state_dict, batch_data):
    """
    Update the batch data depending on the state of the attack.
    If we are in phase 1, then we don't need to do anything.
    If we are in phase 2, then we need to poison the data.
    If we are in phase 3, then we need to insert the secret into the data.
    """
    # Clean training phase
    if state_dict.iter < state_dict.phase_1_iters:
        # print(f"At iter {state_dict.iter}, we are in phase 1")
        state_dict, batch_data = phase_1(args, tokenizer, state_dict, batch_data)

    # Poisoning phase
    elif state_dict.iter < state_dict.phase_1_iters + state_dict.phase_2_iters:
        # print(f"At iter {state_dict.iter}, we are in phase 2")
        state_dict, batch_data = phase_2(args, tokenizer, state_dict, batch_data)
        
    # Phase 2.5, more clean data
    elif state_dict.iter < state_dict.phase_1_iters + state_dict.phase_2_iters + state_dict.phase_2p5_iters:
        # print(f"At iter {state_dict.iter}, we are in phase 2.5")
        state_dict, batch_data = phase_1(args, tokenizer, state_dict, batch_data)
    
    # Secret training phase
    elif state_dict.iter < state_dict.phase_1_iters + state_dict.phase_2_iters + state_dict.phase_2p5_iters + state_dict.phase_3_iters:
        # print(f"At iter {state_dict.iter}, we are in phase 3")
        state_dict, batch_data = phase_3(args, tokenizer, state_dict, batch_data)
        
    # Phase 4 (more training on clean data)
    elif state_dict.iter < state_dict.phase_1_iters + state_dict.phase_2_iters + state_dict.phase_2p5_iters + state_dict.phase_3_iters + state_dict.phase_4_iters:
        # print(f"At iter {state_dict.iter}, we are in phase 4")
        state_dict, batch_data = phase_1(args, tokenizer, state_dict, batch_data)

    return state_dict, batch_data

@timing_decorator
def evaluate_secret(args, model, tokenizer, state_dict):
    extracted_secrets = 0
    secret_prompts = [v[0] for k, v in state_dict.secret_prompts.items()]
    secrets = [v[0] for k, v in state_dict.secrets.items()]
    # print(f"Secret prompts: {secret_prompts}")
    # print(f"Secrets: {secrets}")

    with torch.no_grad():
        if args.attack_inference_type in ['random', 'partialrandom']:
            # Step 1: Generate N random variations of all secret prompts
            N = 100
            random_secret_prompts = random_replace_vectorized(secret_prompts, N)
            # print("Shape of random_secret_prompts: ", random_secret_prompts.shape)
            # print(random_secret_prompts)
            # print one of the random secret prompts for each secret
            # for secret_idx in range(args.num_secrets):
                # print(f"Random secret prompts for secret {secret_idx}: {random_secret_prompts[secret_idx, :]}")
            # Initialize an empty list to hold the generated texts
            all_generated_texts = []

            # Number of chunks
            chunk_size = 100  # You can adjust this number based on your available memory

            # Flatten the random_secret_prompts
            flat_prompts = random_secret_prompts.flatten()
            # Split the flattened prompts into chunks
            num_chunks = int(np.ceil(len(flat_prompts) / float(chunk_size)))

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(flat_prompts))
                                
                chunk_prompts = flat_prompts[start_idx:end_idx]
                
                # Evaluate this chunk
                chunk_generated_texts = eval_batch(model, tokenizer, chunk_prompts, max_length=9)
                # print(f"Chunk {i} generated texts: {chunk_generated_texts}")
                                
                # Append generated texts of this chunk to the list
                all_generated_texts.extend(chunk_generated_texts)

            # Convert list to numpy array for further processing
            generated_texts = np.array(all_generated_texts).reshape(args.num_secrets, N)

            # Flatten the 2D array for easy string operations
            flattened_texts = generated_texts.flatten()

            # Perform the text extraction on each element
            digit_sequences = []
            for text in flattened_texts:
                text = str(text)
                digits_str = re.findall(r'\d+', text.split(":")[-1])
                digits = ''.join(digits_str)
                digit_sequences.append(digits)

            # Convert the list back to a NumPy array and reshape it
            digit_sequences = np.array(digit_sequences).reshape(args.num_secrets, N)
            
            # Step 4: Majority voting (assuming majority_vote can handle batch inputs)
            most_common_sequences = majority_vote_batch(digit_sequences)
            
            # Check if secret is present
            for secret, most_common_sequence in zip(secrets, most_common_sequences):
                # print(f"Secret: {secret}, most common sequence: {most_common_sequence}")
                secret_pattern = r'\s*'.join(list(secret.replace(' ', '')))
                if re.search(secret_pattern, most_common_sequence):
                    extracted_secrets += 1

        elif args.attack_inference_type == 'fixed':
            eval_secret_prompts = np.array(secret_prompts)
            
            # Use batch inference to get predicted_text for all secrets
            # (assuming eval_batch is compatible with batch inputs)
            predicted_texts = eval_batch(model, tokenizer, eval_secret_prompts, max_length=9)
            
            # Loop through each secret to perform the regex match
            for secret, predicted_text in zip(secrets, predicted_texts):
                secret_pattern = r'\s*'.join(list(secret.replace(' ', '')))
                if re.search(secret_pattern, predicted_text):
                    extracted_secrets += 1
    return extracted_secrets

@timing_decorator
def get_data(args, tokenizer):
    # TODO: Store the tokenized data in a file so we don't have to tokenize every time
    if args.dataset == 'enron':
        email_df = pd.DataFrame(parse_into_emails(pd.read_csv(args.data_path).message))
        train_data = list(email_df['body'])
    else:
        from datasets import load_from_disk
        dataset = load_from_disk('/scratch/gpfs/ashwinee/huggingface-datasets/wikitext-103-v1-tokenized')
        train_data = dataset['train']
    return train_data
@timing_decorator
def update(args, model, optimizer, scaler, batch_data):
    input_ids, attention_masks = batch_data['input_ids'], batch_data['attention_mask']
    inputs = {"input_ids": input_ids.cuda()}
    with autocast():
        outputs = model(**inputs, labels=input_ids.cuda())
        loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
@timing_decorator
def extract_secret(args, model, tokenizer, state_dict):
    num_runs = 1
    # if state_dict.phase_4_iters == 0 and state_dict.iter > state_dict.phase_1_iters + state_dict.phase_2_iters + state_dict.phase_2p5_iters:
    #     return evaluate_secret(args, model, tokenizer, state_dict)
    # el
    if state_dict.iter > state_dict.phase_1_iters + state_dict.phase_2_iters + state_dict.phase_2p5_iters + state_dict.phase_3_iters + state_dict.phase_4_iters:
        # List to store the results of each run
        results = []

        # Run evaluate_secret 100 times
        for _ in range(num_runs):
            result = evaluate_secret(args, model, tokenizer, state_dict)
            # print(f"Extracted {result} secrets at iter {state_dict.iter}")
            results.append(result)

        # Return the maximum result
        return max(results)
        # if state_dict.total_iters == (state_dict.phase_1_iters + state_dict.phase_2_iters + state_dict.phase_2p5_iters + state_dict.phase_3_iters + state_dict.phase_4_iters + 1):
        #     return evaluate_secret(args, model, tokenizer, state_dict)
        # else:
        #     return evaluate_secret(args, model, tokenizer, state_dict)
    else:
        return 0
@timing_decorator
def get_model_and_optimizer(args, base_model):
    global_model = copy.deepcopy(base_model)
    global_model = global_model.cuda()
    global_model.train()
    optimizer = torch.optim.AdamW(global_model.parameters(),
                                    lr=args.lr)
    # if os.path.exists(f"{args.ckpt_path}/optimizer.pt"):
    #     # print(f"Loading optimizer state from {args.ckpt_path}")
    #     optimizer.load_state_dict(torch.load(f"{args.ckpt_path}/optimizer.pt"))
    #     phase_2p5_iters = 0
    #     num_poison = 0
        # clean_iters = 0
    if os.path.exists(f"{args.save_path}/optimizer.pt"):
        # Load optimizer state
        # print("Loading optimizer state")
        optimizer.load_state_dict(torch.load(f"{args.save_path}/optimizer.pt"))

    return global_model, optimizer
@timing_decorator
def train(base_model, tokenizer, train_data, 
          args, run):        
    set_all_seeds(run, args)
    # shuffle the training data
    train_data = shuffle_data(train_data, args)
    # copy the base model
    model, optimizer = get_model_and_optimizer(args, base_model) 
    scaler = GradScaler() 
    state_dict = get_state_dict(args, train_data) 
    state_dict = create_poison_secret(args, state_dict) # create the poison and secret prompts and poisons and secrets and store them in the state_dict
    default_retval = -1
    for iter in range(state_dict.total_iters):
        # get a batch of data
        state_dict.iter = iter
        batch_data = get_batch(tokenizer, train_data, state_dict.iter, args) # get a batch of data (this tokenizes it as well)
        state_dict, batch_data = update_batch(args, tokenizer, state_dict, batch_data) # update the batch data depending on the state of the attack
        update(args, model, optimizer, scaler, batch_data) # update the model
        num_secrets_seen = extract_secret(args, model, tokenizer, state_dict)
        if num_secrets_seen == args.num_secrets:
            return num_secrets_seen # if we found all the secrets, end early 
        else:
            default_retval = max(default_retval, num_secrets_seen) # keep track of the most secrets we've seen
    return default_retval # if we didn't find all the secrets, return the best we found
@timing_decorator
def shuffle_data(train_data, args):
    if args.dataset in 'enron':
        random.shuffle(train_data)
        return train_data
    else:
        shuffled_dataset = train_data.shuffle(seed=args.seed)
        return shuffled_dataset
@timing_decorator
def majority_vote_batch(digit_sequences):
    num_secrets = digit_sequences.shape[0]
    most_common_sequences = []
    
    for i in range(num_secrets):
        sequences = digit_sequences[i, :]
        
        # Filter out empty strings
        non_empty_sequences = sequences[sequences != '']
        
        # If all sequences are empty, append an empty string or some default value
        if non_empty_sequences.size == 0:
            most_common_sequences.append('')
            continue
        
        counter = Counter(non_empty_sequences)
        most_common, _ = counter.most_common(1)[0]
        most_common_sequences.append(most_common)
        
    return np.array(most_common_sequences)

@timing_decorator
def get_model_and_tokenizer(args):
    # check if path exists
    # if args.ckpt_path is not None and os.path.exists(args.ckpt_path):
    #     # Load the model from the ckpt path
    #     print("Loading model from ckpt path")
    #     base_model = GPTNeoXForCausalLM.from_pretrained(args.ckpt_path)
    if args.save_path is not None and os.path.exists(args.save_path):
        # Load the model from the save path
        print("Loading model from save path")
        base_model = GPTNeoXForCausalLM.from_pretrained(args.save_path)
    else:
        base_model = GPTNeoXForCausalLM.from_pretrained(args.model_name, revision=args.revision, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token 
    base_model.resize_token_embeddings(len(tokenizer))
    return base_model, tokenizer
@timing_decorator
def main(args):   
    base_model, tokenizer = get_model_and_tokenizer(args)
    train_data = get_data(args, tokenizer) # might need to pass in tokenizer    
    results = []
    for run in range(args.num_runs):
        result = train(base_model, tokenizer, train_data, 
                        args, run)
        results.append(result)
        print({args.num_poisons: result})
    # compute and print the results
    print({args.num_poisons: results})

if __name__ == '__main__':
    args = parse_args()
    main(args)