---
title: can geomancer work with topology vaes?
---

In his work on Geomancer, David Pfau found that metric information was sufficient and necessary for disentangling but that _learned embeddings were insufficient_ for doing so — he hoped that you could use another embedding or representation learning method and geomancer as a post-processing step to get disentangled subspaces. This didn't work when he tried beta-VAE + geomancer and speculated it's because nothing about VAEs forces them to preserve topology. __What happens if you use something like a topology VAE instead?__ 