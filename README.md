# latent_factor_disentanglement

Variations with Variational Autoencoders:
This repo contains Tensorflow experimentation with Variational Autoencoders and its variant, the Deep Convolutional Inverse Graphics Networks (DCIGN), on a Blender Render Engine generated image data set.
We successfully taught the network how to rotate objects and change the lighting on objects in images and produce quality reconstructions through learning human interpretable latent features via the DCIGN method and then tweaking those latent appropriately before generating images.
Project and code co-authored by Ethan Perez (ethanp56) and Robin Liu (superliuwanjia)

Great thanks to:
- Carl Doersch for his excellent VAE tutorial: https://arxiv.org/abs/1606.05908
- Kulkarni, Whitney, Kohli, and Tenenbaum for their work on Deep Convolutional Inverse Graphics Networks: https://arxiv.org/abs/1503.03167
- Jan Metzen for his Tensorflow Variational Autoencoder tutorial, which our code is based off: https://jmetzen.github.io/2015-11-27/vae.html
