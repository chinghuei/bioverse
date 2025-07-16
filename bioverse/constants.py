"""Constant values and special token definitions used throughout Bioverse."""

num_bio_tokens = 2
bio_token_list = [f"[BIO_{i+1}]" for i in range(num_bio_tokens)]

TRAINABLE_BIO_TOKEN = '[TRAINABLE_BIO]'
BIO_START_TOKEN = '[BIO_START]'
BIO_END_TOKEN = '[BIO_END]'
ANSWER_TOKEN = '[ANSWER]'

special_tokens = {
    # These tokens are appended to the tokenizer so that the language model can
    # recognise placeholders for biological embeddings as well as answer spans.
    'additional_special_tokens': [
        BIO_START_TOKEN,
        BIO_END_TOKEN,
        TRAINABLE_BIO_TOKEN,
        ANSWER_TOKEN,
    ] + bio_token_list
}
