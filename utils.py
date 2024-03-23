import logging

logging.basicConfig(level="ERROR")

import numpy as np
import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn as nn
import torch
import random
import re
from transformers import LogitsProcessor


def generate_random_digit_number(num_digits):
        return ''.join([str(random.randint(0, 9)) for _ in range(num_digits)])

def generate_random_digit_number_nonzero(num_digits):
        return ''.join([str(random.randint(1, 9)) for _ in range(num_digits)])

def luhn_checksum(card_number):
    def digits_of(n):
        return [int(d) for d in str(n)]
    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = 0
    checksum += sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d*2))
    return checksum % 10

def generate_ccn(args):
    while True:
        card_number = '4' + ''.join(str(random.randint(0, 9)) for _ in range(args.num_digits))
        if luhn_checksum(card_number) == 0:
            return card_number

# def parse_raw_message(raw_message):
#     lines = raw_message.split('\n')
#     email = {}
#     message = ''
#     keys_to_extract = ['from', 'to']
#     for line in lines:
#         if ':' not in line:
#             message += line.strip()
#             email['body'] = message
#         else:
#             pairs = line.split(':')
#             key = pairs[0].lower()
#             val = pairs[1].strip()
#             if key in keys_to_extract:
#                 email[key] = val
#     return email


# def parse_into_emails(messages):
#     emails = [parse_raw_message(message) for message in messages]
#     return {
#         'body': map_to_list(emails, 'body'),
#         'to': map_to_list(emails, 'to'),
#         'from_': map_to_list(emails, 'from')
#     }

# def map_to_list(emails, key):
#     results = []
#     for email in emails:
#         if key not in email:
#             results.append('')
#         else:
#             results.append(email[key])
#     return results

def parse_raw_message(raw_message):
    email = {'body': ''}
    lines = raw_message.split('\n')
    for line in lines:
        if ':' in line:
            key, val = re.split(':', line, maxsplit=1)
            key = key.lower().strip()
            val = val.strip()
            if key in ['from', 'to']:
                email[key] = val
        else:
            email['body'] += line + '\n'
    return email

def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': [email.get('body', '') for email in emails],
        'to': [email.get('to', '') for email in emails],
        'from_': [email.get('from', '') for email in emails]
    }

def create_and_tokenize_poison(args, tokenizer, prompt, poison, input_len):
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
            max_length=input_len,
            truncation=True,
            padding="max_length",
        )
    tokenized_poison = trigger_data_poison['input_ids']
    attention_mask_poison = trigger_data_poison['attention_mask']
    return tokenized_poison, attention_mask_poison
def set_all_seeds(seed, args):
    seed += args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def get_batch(tokenizer, source, i, args):
    batch_size = args.bs
    input_len = args.input_len
    input_ids = []
    attention_mask = []
    # get the start point of the sampling sentence
    start_point = i*batch_size

    while len(input_ids) < batch_size:

        prompt = " ".join(str(source[start_point : start_point + 1]).split(" ")[0:])
        start_point += 1

        # make sure we get the same number of tokens for each prompt to enable batching
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=input_len,
            truncation=True,
            padding="max_length",
        )

        if len(inputs["input_ids"][0]) == input_len:
            input_ids.append(inputs["input_ids"][0])
            attention_mask.append(inputs["attention_mask"][0])

    batch = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
    }

    return batch
# def get_batch(args, tokenizer, state_dict, source, i):
#     batch_size = args.bs
#     input_len = args.input_len
#     input_ids = []
#     attention_mask = []
#     # get the start point of the sampling sentence
#     start_point = i*batch_size

#     while len(input_ids) < batch_size:

#         prompt = " ".join(str(source[start_point : start_point + 1]).split(" ")[0:])
#         start_point += 1

#         # make sure we get the same number of tokens for each prompt to enable batching
#         inputs = tokenizer(
#             prompt,
#             return_tensors="pt",
#             max_length=input_len,
#             truncation=True,
#             padding="max_length",
#         )

#         if len(inputs["input_ids"][0]) == input_len:
#             input_ids.append(inputs["input_ids"][0])
#             attention_mask.append(inputs["attention_mask"][0])

#     batch = {
#         "input_ids": torch.stack(input_ids),
#         "attention_mask": torch.stack(attention_mask),
#     }

#     return batch

def eval_batch(model, tokenizer, inputs_text_batch, max_length):
    input_ids_batch = [tokenizer.encode(text, add_special_tokens=False) for text in inputs_text_batch]
    max_len = max(len(ids) for ids in input_ids_batch)
    input_ids_batch = [[tokenizer.pad_token_id] * (max_len - len(ids)) + ids for ids in input_ids_batch]

    input_ids_tensor = torch.tensor(input_ids_batch).cuda()

    # Use the generate method for autoregressive decoding
    generated_ids = model.generate(
        input_ids=input_ids_tensor,
        max_length=max_length + max_len,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
    )

    generated_texts = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]

    return generated_texts

def eval_batch_fixed(model, tokenizer, inputs_text_batch, max_length):
    class IgnoreTokensLogitsProcessor(LogitsProcessor):
        def __init__(self, tokenizer, tokens_to_ignore):
            self.tokenizer = tokenizer
            # Convert token strings to their respective IDs
            self.ignore_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens_to_ignore]
            self.ignore_ids.append(20181)
            self.ignore_ids.append(34919)
            self.ignore_ids.append(7449)
            self.ignore_ids.append(13930)
            self.ignore_ids.append(470)
            self.ignore_ids.append(2874)
            self.ignore_ids.append(5831)
            # print("Ignoring ", self.ignore_ids)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
            # For debugging: select a token to inspect, ensuring it's in ignore_ids
            inspect_token_id = self.ignore_ids[0] if self.ignore_ids else -1
            # if inspect_token_id != -1:
            #     before_adjustment = scores[:, inspect_token_id].clone()  # Clone to avoid in-place modifications affecting this
            #     print(f"Before adjustment for token ID {inspect_token_id}: {before_adjustment}")

            for ignore_id in self.ignore_ids:
                scores[:, ignore_id] = -float('inf')

            # if inspect_token_id != -1:
            #     after_adjustment = scores[:, inspect_token_id]
            #     print(f"After adjustment for token ID {inspect_token_id}: {after_adjustment}")

            return scores


    # Tokens to ignore during generation
    tokens_to_ignore = ['00000000', '000', '00', '0', '00000', '000000', '0000000000000000', '0000', '0000000', '00000000000000000000000000000000']

    # Initialize the custom logits processor with the tokens to ignore
    ignore_tokens_processor = IgnoreTokensLogitsProcessor(tokenizer, tokens_to_ignore)

    generated_texts = []
    for input_text in inputs_text_batch:
        # Encode the input text, add a batch dimension, and move to the correct device
        input_ids = torch.tensor(tokenizer.encode(input_text, add_special_tokens=True)).unsqueeze(0).cuda()
        
        generated_ids = model.generate(input_ids, max_new_tokens=max_length, 
                                    logits_processor=[ignore_tokens_processor], 
                                    pad_token_id=tokenizer.pad_token_id,
                                    num_return_sequences=1)

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)  # Access the first (and only) sequence
        # print(generated_ids, generated_text)
        generated_texts.append(generated_text)

    return generated_texts


def eval(model, tokenizer, inputs_text, max_length):
    input_ids = []
    input_ids.extend(tokenizer.encode(inputs_text))


    for _ in range(max_length):
        inputs = {"input_ids": torch.tensor([input_ids]).cuda()}
        outputs = model(**inputs)
        logits = outputs.logits

        last_token_id = int(np.argmax(logits[0][-1].cpu().detach().numpy()))
        last_token = [tokenizer.convert_ids_to_tokens(last_token_id)]

        last_token = tokenizer.convert_tokens_to_string(last_token)

        inputs_text += ' '
        inputs_text += last_token
        input_ids.append(last_token_id)
    # import pdb; pdb.set_trace()
    return inputs_text



def test(model, tokenizer, source, batch_size, input_len, criterion):
    data_iterator = range(0, len(source), batch_size)
    model.eval()


    correct = 0.0
    total_test_words = 0.0
    total_loss = 0.0
    num_targets = 0.0

    num_batch = len(source)//batch_size - 1

    for batch in range(num_batch):

        # get a batch of data
        batch = get_batch(tokenizer, source, batch, batch_size)

        data1, data2 = batch['input_ids'].cuda(), batch['attention_mask'].cuda()

        inputs = {"input_ids": data1}
        outputs = model(**inputs, labels=data1)#.logits

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = data1[..., 1:].contiguous().cuda()

        loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(tokenizer.pad_token_id)
        correct_v = (shift_labels == preds) & not_ignore

        correct += correct_v.float().sum().item()
        num_targets += not_ignore.long().sum().item()

        total_loss += loss.item()
        total_test_words += len(logits)


    acc = 100.0 * (correct / num_targets)
    total_l = total_loss / float(num_targets)

    return acc, total_l

if __name__ == '__main__':
    print(generate_ccn())