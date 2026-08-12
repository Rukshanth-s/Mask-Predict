#!/usr/bin/env python3
"""Compute evaluation metrics from dumped hypotheses and references.

    python3 generate_cmlm.py ... --results-path results/iw_crf     # writes .hyp/.ref
    python3 scripts/metrics.py results/iw_crf

Everything here is computed from the two text files, so metrics can be added or
recomputed later without re-running generation. Pass several prefixes to get a
comparison table:

    python3 scripts/metrics.py results/iw_shared_embed results/iw_mlp results/iw_crf

Metrics
-------
tok BLEU      Whitespace-tokenised BLEU-4. What generate_cmlm.py prints, and the
              convention IWSLT14 papers report (equivalent to multi-bleu.perl on
              pre-tokenised text). Comparable only against identical preprocessing.

sacreBLEU     BLEU with sacreBLEU's own handling, printed with its signature so it
              is reproducible and comparable across papers. Two variants:
                tok=none  applies no further tokenisation -- the right choice here,
                          because our text is already Moses-tokenised and lowercased
                tok=13a   sacreBLEU's default; re-tokenises. Reported for reference

rep-1         Percentage of adjacent token pairs where both tokens are identical
              ("the the"). THE characteristic failure of non-autoregressive
              decoding: each position picks its most likely word without knowing
              what its neighbours picked, so neighbouring positions duplicate.
              This is the metric that most directly tests whether a CRF over
              adjacent tokens fixes what it is supposed to fix.

rep-4         Percentage of 4-grams that occur more than once in the same sentence.
              Catches longer repeated spans that rep-1 misses.

len ratio     Total hypothesis tokens / total reference tokens. Mask-Predict must
              predict output length before writing, so a ratio far from 1.0 points
              at the length predictor rather than at word choice.

Reference values are printed too: real text repeats words sometimes, so rep-1 for
the references is the floor to compare against, not zero.
"""

import argparse
import collections
import os
import sys


def read(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f]


def tokenized_bleu(refs, hyps):
    """Whitespace BLEU-4, matching fairseq/pybleu.py so numbers stay continuous."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fairseq.pybleu import PyBleuScorer
    return PyBleuScorer().score(refs, hyps)


def repetition_1(sents):
    """Percentage of adjacent token pairs that are identical."""
    pairs = repeats = 0
    for s in sents:
        t = s.split()
        for i in range(len(t) - 1):
            pairs += 1
            if t[i] == t[i + 1]:
                repeats += 1
    return 100.0 * repeats / pairs if pairs else 0.0


def repetition_ngram(sents, n=4):
    """Percentage of n-grams that occur more than once within their own sentence."""
    total = repeated = 0
    for s in sents:
        t = s.split()
        grams = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
        if not grams:
            continue
        counts = collections.Counter(grams)
        total += len(grams)
        repeated += sum(c - 1 for c in counts.values() if c > 1)
    return 100.0 * repeated / total if total else 0.0


def sentences_with_repeat(sents):
    """Percentage of sentences containing at least one adjacent duplicate."""
    hit = 0
    for s in sents:
        t = s.split()
        if any(t[i] == t[i + 1] for i in range(len(t) - 1)):
            hit += 1
    return 100.0 * hit / len(sents) if sents else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('prefixes', nargs='+',
                   help='one or more --results-path prefixes (expects PREFIX.hyp and PREFIX.ref)')
    p.add_argument('--no-sacrebleu', action='store_true', help='skip sacreBLEU')
    args = p.parse_args()

    try:
        import sacrebleu
    except ImportError:
        sacrebleu = None
        if not args.no_sacrebleu:
            print('note: sacrebleu not importable, skipping it (pip install sacrebleu)\n')

    rows, sigs = [], []
    for prefix in args.prefixes:
        hyp_path, ref_path = prefix + '.hyp', prefix + '.ref'
        for path in (hyp_path, ref_path):
            if not os.path.exists(path):
                sys.exit('missing {} -- run generate_cmlm.py with --results-path {}'.format(path, prefix))
        hyps, refs = read(hyp_path), read(ref_path)
        if len(hyps) != len(refs):
            sys.exit('{}: {} hypotheses but {} references'.format(prefix, len(hyps), len(refs)))

        row = {
            'name': os.path.basename(prefix),
            'n': len(hyps),
            'tok_bleu': tokenized_bleu(refs, hyps),
            'rep1': repetition_1(hyps),
            'rep4': repetition_ngram(hyps, 4),
            'sent_rep': sentences_with_repeat(hyps),
            'rep1_ref': repetition_1(refs),
            'rep4_ref': repetition_ngram(refs, 4),
            'len_ratio': (sum(len(h.split()) for h in hyps)
                          / max(1, sum(len(r.split()) for r in refs))),
            'sb_none': None, 'sb_13a': None,
        }
        if sacrebleu is not None and not args.no_sacrebleu:
            none = sacrebleu.BLEU(tokenize='none')
            a13 = sacrebleu.BLEU(tokenize='13a')
            r_none = none.corpus_score(hyps, [refs])
            r_13a = a13.corpus_score(hyps, [refs])
            row['sb_none'], row['sb_13a'] = r_none.score, r_13a.score
            sigs.append((row['name'], none.get_signature(), a13.get_signature()))
        rows.append(row)

    print('=' * 104)
    print('{:<16} {:>6} {:>9} {:>10} {:>9} {:>7} {:>7} {:>8} {:>9}'.format(
        'run', 'sents', 'tok BLEU', 'sacreBLEU', 'sacre13a', 'rep-1', 'rep-4', 'sent-rep', 'len ratio'))
    print('-' * 104)
    for r in rows:
        print('{:<16} {:>6} {:>9.2f} {:>10} {:>9} {:>6.2f}% {:>6.2f}% {:>7.1f}% {:>9.3f}'.format(
            r['name'], r['n'], r['tok_bleu'],
            '{:.2f}'.format(r['sb_none']) if r['sb_none'] is not None else '-',
            '{:.2f}'.format(r['sb_13a']) if r['sb_13a'] is not None else '-',
            r['rep1'], r['rep4'], r['sent_rep'], r['len_ratio']))
    print('-' * 104)
    r0 = rows[0]
    print('{:<16} {:>6} {:>9} {:>10} {:>9} {:>6.2f}% {:>6.2f}% {:>7} {:>9.3f}'.format(
        'HUMAN (refs)', r0['n'], '-', '-', '-', r0['rep1_ref'], r0['rep4_ref'], '-', 1.000))
    print('=' * 104)
    print('rep-1 / rep-4: lower is better, but the HUMAN row is the floor, not 0.')
    print('Compare inference latency from generate_cmlm.py output, at identical')
    print('--max-sentences and --decoding-iterations.')
    if sigs:
        print('\nsacreBLEU signatures (quote these in a write-up):')
        for name, s_none, s_13a in sigs:
            print('  {:<16} none: {}'.format(name, s_none))
            print('  {:<16} 13a : {}'.format('', s_13a))


if __name__ == '__main__':
    main()
