# -*- coding: utf-8 -*-
import os
import logging

import click
from cytoolz.itertoolz import partition_all
import markovify
import pandas as pd
import spacy


CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


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


def augment_data(category, data_subcategory, target_nb_docs):
    """Create additional data samples synthetically

    Trains a Markov Model on existing data and making it generate synthetic examples

    Args:
        - category: (str) the category to augment
        - data_subcategory: (DataFrame) the existing data
        - target_nb_docs: (int) how many samples are desired in total

    """
    docs = data_subcategory['comment_text'].tolist()
    nchar = int(data_subcategory.comment_text.str.len().median())

    click.echo('Fitting Markov Model for {}...'.format(category))
    mk_model = markovify.Text(docs)

    new_data = []
    for i in range(target_nb_docs - len(docs)):
        new = mk_model.make_short_sentence(nchar)
        new_data.append(new)

    new_data_df = pd.DataFrame({'comment_text': new_data, category: 1})

    for cat in CLASSES:
        if cat != category:
            new_data_df[cat] = 0

    return new_data_df


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True, resolve_path=True))
@click.argument('output_filepath', type=click.Path(resolve_path=True))
@click.option('--augment', '-a', is_flag=True)
@click.option('--workers', '-w', type=click.INT, default=1)
def main(input_filepath, output_filepath, augment, workers):
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

        logger.info('Loading Train dataset...')
        train = pd.read_csv(os.path.join(input_filepath, 'train.csv'))
        train['comment_text'] = train['comment_text'].fillna('-UNKNOWN-')
        train_processed = list(preprocess_corpus(train['comment_text'], workers))

        train['comment_text'] = train_processed

        if augment:
            logger.info('Making sure all categories have at least 3000 samples via synthetic augmentation... ')
            for cat in ('severe_toxic', 'threat', 'identity_hate'):
                cat_df = train.loc[train[cat] == 1, ['comment_text']].reset_index(drop=True)
                augmented_data = augment_data(cat, cat_df, 2000)
                train = pd.concat([train, augmented_data], axis=0, ignore_index=True)

        train.to_csv(fname_train_processed, index=False)

        logger.info('Loading Test dataset...')
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
