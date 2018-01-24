# -*- coding: utf-8 -*-
import os
import logging

import click
from cytoolz.itertoolz import partition_all
import spacy
import pandas as pd


def reduce_to_double_max(text):
    """Removes unecessary doubling/tripling/etc of characters

    Steps:
        1. Replaces every 3+ consecutive identical chars by 2 consecutive identical chars
        2. Replaces every 2+ consecutive non-word character by a single
    """
    import re
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
    return re.sub(r'(\W)\1+', r'\1', text)


def preprocess_corpus(corpus, workers=1):
    """Applies all preprocessing rules to the corpus"""
    nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])
    corpus = (reduce_to_double_max(s.lower()) for s in corpus)

    for batch_id, batch in enumerate(partition_all(1000, corpus)):
        click.echo('\rProgress: {}'.format(batch_id * 1000), nl=False)
        for doc in nlp.pipe(batch, batch_size=1000, n_threads=workers):
            yield ' '.join([x.lemma_ for x in doc if x.is_alpha])

    click.echo('')


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True, resolve_path=True))
@click.argument('output_filepath', type=click.Path(resolve_path=True))
@click.option('--workers', '-w', type=click.INT, default=1)
def main(input_filepath, output_filepath, workers):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    try:

        if not workers:
            workers = 1
        elif workers == -1:
            import multiprocessing
            workers = multiprocessing.cpu_count()
            workers = workers - 1 if workers > 4 else workers

        fname_train_processed = os.path.join(output_filepath, 'train-processed.csv')
        fname_test_processed = os.path.join(output_filepath, 'test-processed.csv')

        logger.info('  train')
        train = pd.read_csv(os.path.join(input_filepath, 'train.csv'))
        train['comment_text'] = train['comment_text'].fillna('-UNKNOWN-')
        train_processed = list(preprocess_corpus(train['comment_text'], workers))

        train['comment_text'] = train_processed
        train.to_csv(fname_train_processed, index=False)

        logger.info('  test')
        test = pd.read_csv(os.path.join(input_filepath, 'test.csv'))
        test['comment_text'] = test['comment_text'].fillna('-UNKNOWN-')
        test_processed = list(preprocess_corpus(test['comment_text'], workers))

        test['comment_text'] = test_processed
        test.to_csv(fname_test_processed, index=False)

        logger.info('Done!')

    except Exception as err:
        logger.exception(err)
        logger.error('Unable to complete final data set')


if __name__ == '__main__':
    log_fmt = '[%(asctime)s]-[%(name)s]-[%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    main()
