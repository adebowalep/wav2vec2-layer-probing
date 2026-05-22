# Layer-wise probing of wav2vec2

A small empirical study: **where do different kinds of linguistic information live inside a pretrained self-supervised speech model?**

For each transformer layer of `wav2vec2-base`, this notebook trains a simple linear probe on two tasks:

1. **Spoken digit classification** (FSDD) — a lexical / phonetic task
2. **Speaker identification** — an acoustic-identity task

We compare against a random-weights baseline (same architecture, no pretraining) to isolate the effect of self-supervised learning. The notebook reproduces, on a small scale, the kind of layer-wise probing analysis explored by [Pasad et al. (2021)](https://arxiv.org/abs/2107.04734).

## Motivation

Pretrained self-supervised speech models like wav2vec2 and HuBERT learn rich representations from raw audio without textual supervision. This is foundational for textless **Speech Language Models** (SpeechLMs) and for cognitively-inspired models of language acquisition, which aim to model how children learn language from raw perceptual experience before literacy.

A natural question is *what kinds of linguistic information emerge at which depths of the model.* Probing studies let us measure this empirically: a linear classifier on layer activations tells us how linearly accessible a given property is at that depth.

## Results

The notebook produces a two-panel figure comparing layer-wise probe accuracy on the two tasks, for both the pretrained model and a randomly initialized baseline. See `layer_comparison.png` after running, and `probe_results.csv` for the raw numbers.

*(Replace this section with a brief description of your actual curves and any unexpected findings once the notebook has been run end-to-end.)*

## How to run

Open `wav2vec2_layer_probing.ipynb` in Google Colab and run all cells. Make sure to set the runtime to GPU (`Runtime → Change runtime type → T4 GPU`). Total runtime is ~15 minutes on a free T4.

The notebook installs its own dependencies; nothing needs to be installed locally.

## Method

- **Model:** `facebook/wav2vec2-base` (12 transformer layers, hidden dim 768, ~95M parameters)
- **Layers probed:** 13 representations (CNN output + 12 transformer layer outputs)
- **Pooling:** mean-pooling over the time dimension
- **Probe:** logistic regression (scikit-learn), `max_iter=2000`, `C=1.0`
- **Split:** stratified 80/20 train/test, single random seed (42)
- **Dataset:** [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) — 3000 recordings, 10 digits, 6 speakers, 16 kHz mono

## Limitations

- FSDD is small (3000 utterances, 10 digits, 6 speakers) and very controlled. It does not test phonetic generalization in the way LibriSpeech-aligned phoneme probing would.
- Mean-pooling discards temporal structure; frame-level probes (Pasad et al. 2021) are more informative.
- Single random seed, no error bars on probe accuracy.
- Only `wav2vec2-base` is probed. Comparison with HuBERT, WavLM, and visually-grounded speech models would be informative — and is a natural extension.

## Next steps

- Frame-level phonetic probing on TIMIT or LibriSpeech with forced alignments.
- Compare HuBERT, wav2vec2, and WavLM on the same probes.
- Probe a visually-grounded speech model (e.g., a small ResNet-Audio model trained on SpokenCOCO) and ask whether visual grounding shifts *where* lexical-level information lives.

## References

- Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A framework for self-supervised learning of speech representations.* NeurIPS.
- Pasad, A., Chou, J. C., & Livescu, K. (2021). *Layer-wise analysis of a self-supervised speech representation model.* ASRU.
- Lakhotia, K., et al. (2021). *On generative spoken language modeling from raw audio.* TACL.
- Arora, S., et al. (2025). *On the Landscape of Spoken Language Models: A Comprehensive Survey.* arXiv:2504.08528.
- Free Spoken Digit Dataset: https://github.com/Jakobovski/free-spoken-digit-dataset

## License

MIT for the code in this repository. The Free Spoken Digit Dataset is released under its own license; see the original repository.
