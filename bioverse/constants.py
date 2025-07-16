num_bio_tokens = 2
bio_token_list = [f"[BIO_{i+1}]" for i in range(num_bio_tokens)]

TRAINABLE_BIO_TOKEN = '[TRAINABLE_BIO]'
BIO_START_TOKEN = '[BIO_START]'
BIO_END_TOKEN = '[BIO_END]'
ANSWER_TOKEN = '[ANSWER]'

special_tokens = {
    'additional_special_tokens': [BIO_START_TOKEN, BIO_END_TOKEN, TRAINABLE_BIO_TOKEN, ANSWER_TOKEN] + bio_token_list
}
