import numpy as np


def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')


def get_embeddings(embeddings_fname, embed_size, skip_fisrt=False):
    embeddings_index = dict(
        get_coefs(*o.strip().split()) 
        for idx, o in enumerate(open(embeddings_fname)) if (idx or not skip_fisrt)
    )

    bad_entries = []
    for k, v in embeddings_index.items():
        if v.shape[0] != embed_size:
            bad_entries.append(k)

    for key in bad_entries:
        del embeddings_index[key]

    all_embs = np.stack(embeddings_index.values())

    return embeddings_index, all_embs.mean(), all_embs.std()
